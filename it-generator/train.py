import argparse
import logging
import os
import time
import random
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from transformers.models.lxmert.modeling_lxmert import LxmertConfig
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule
from modeling import Generator

from config import _C as config
from dataset import COCOCaptionDataset, collate_fn_train

from loss import LabelSmoothingLoss
from utils import get_rank, mkdir, synchronize
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger
from inference import test
from utils.tokenizer import get_tokenizer
import re

import numpy as np

from sample_images import sample_image
from sample_captions import sample_caption
from loss import get_self_critical_reward, RewardCriterion


def train(generator, optimizer, data_loader, scheduler, checkpointer,
          device, arguments, clip_scorer, save_dir, config):
    logger = logging.getLogger("train")
    logger.info("Start training")
    max_iter = len(data_loader)
    start_iter = arguments['iteration']
    generator.train()

    tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)
    log_time = config.log_time
    checkpoint_time = config.checkpoint_time

    if config.fp16:
        from apex import amp


    criterion = LabelSmoothingLoss(num_tokens)
    mse_criterion = torch.nn.MSELoss()
    mgcluster_criterion = LabelSmoothingLoss(config.grid_cluster_num)
    rl_crit = RewardCriterion()

    end = time.time()
    optimizer.zero_grad()

    task_dict = {
        'xe_i2t': 0,
        'xe_t2i': 1,
        'rl_i2t': 0,
        'rl_t2i': 1
    }

    use_discrete = config.use_grid_cluster and not config.use_grid_feat

    for iteration, batch in enumerate(data_loader, start_iter):

        iteration = iteration + 1
        arguments['iteration'] = iteration

        task = config.iteration_tasks[iteration%len(config.iteration_tasks)]
        iteration_t2i = task_dict[task]

        if config.use_mix_feature:
            if iteration_t2i:
                use_discrete = True
            else:
                use_discrete = False

        if use_discrete:
            cluster_index = batch[0].to(device)
            grid_features = None
        else:
            cluster_index = None
            grid_features = batch[3].to(device)  # (N, 64, 2048), float

        input_token_ids = batch[1].to(device)  # (N, L), long
        masked_token_ids = batch[2].to(device)  # (N, L), long

        batch_size = input_token_ids.size(0)
        visual_token_num = config.grid_size * config.grid_size

        if iteration_t2i:  # attend to img
            # for captions
            attention_mask = torch.ones(input_token_ids.shape, dtype=torch.float32, device=device)
            attention_mask[input_token_ids == PAD] = 0  # do not attend to PAD

            # for image
            _visual_mask = torch.zeros((batch_size, visual_token_num), dtype=torch.float32, device=device)
            # need to mask token content in selected_idx for prediction/generation
            num_masks = random.randint(max(1, int(0.1 * visual_token_num)), visual_token_num)
            selected_idx = random.sample(range(visual_token_num), num_masks)
            _visual_mask[:, selected_idx] = 1
            mask_position = (_visual_mask == 1).to(torch.long).view(-1)
            mask_position = mask_position.nonzero().squeeze()

        else:  # mask text
            attention_mask = torch.ones(masked_token_ids.shape, dtype=torch.float32, device=device)
            for i in range(len(attention_mask)):
                mask_idx = torch.where(masked_token_ids[i] == MASK)[0][0]  # position of [MASK]
                # the end idx of this caption (including all [eos])
                end_idx = torch.where(attention_mask[i] == 1)[0][-1]
                attention_mask[i, mask_idx + 1:end_idx + 1] = 0  # do not attend to tokens after [MASK]

        _attention_mask = torch.ones((batch_size, visual_token_num), dtype=torch.float32, device=device)
        attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

        loss = 0
        image_mse_loss = 0
        clip_score = torch.zeros((2,3), dtype=torch.float)  # anything sum to zero
        reward = torch.zeros((2,3), dtype=torch.float)  # anything sum to zero
        scst_reward = []

        if task == 'rl_t2i':
            generator.train()
            predict_indices, code_logit, pred_scores = sample_image(
                sentences=None, generator=generator, config=config,
                input_token_ids=input_token_ids, attention_mask=attention_mask)

            text_pred_scores, img_pred_scores = pred_scores
            loss += 0 * torch.sum(img_pred_scores)
            loss += 0 * torch.sum(text_pred_scores)

            gumbel_soft = F.gumbel_softmax(code_logit, dim=-1, hard=False)  # (bsz,8x8,10000)
            # (bsz,8x8,10000)x(10000,2048)->(bsz,8x8,2048)
            visual_features = torch.matmul(gumbel_soft, generator.module.cluster_codebook.weight)

            if config.mse_loss:
                gt_grid_features = generator.module.cluster_codebook(cluster_index)
                image_mse_loss = mse_criterion(gt_grid_features, visual_features)
                loss += image_mse_loss * config.mse_loss

            if config.use_clip_score:
                real_sents = [tokenizer.decode(sent) for sent in input_token_ids.tolist()]
                visual_features = visual_features.view(len(real_sents), config.grid_size, config.grid_size, -1)
                visual_features = visual_features.permute(0, 3, 1, 2)
                recon_img = generator.module.GAN.G(visual_features)

                clip_score = clip_scorer(recon_img, real_sents)
                loss += - torch.mean(clip_score)

        elif task == 'rl_i2t':
        # self-critical text generation
            generator.eval()
            greedy_res, _, _ = sample_caption(visual_features=grid_features,
                                                  generator=generator, tokenizer=tokenizer,
                                                  config=config,
                                                  cluster_index=cluster_index,
                                                  sample_mode='greedy',
                                                  masked_token_ids=masked_token_ids,
                                                  attention_mask=attention_mask)
            generator.train()
            gen_result, sample_logprobs, pred_scores = sample_caption(visual_features=grid_features,
                                                  generator=generator, tokenizer=tokenizer,
                                                  config=config,
                                                  cluster_index=cluster_index,
                                                  sample_mode='sample',
                                                  masked_token_ids=masked_token_ids,
                                                  attention_mask=attention_mask,
                                                  return_res_prob = True)
            # [tokenizer.decode(res.cpu().numpy(), end_flags=[EOS]) for res in greedy_res]
            # [tokenizer.decode(res.cpu().numpy(), end_flags=[EOS]) for res in gen_result]
            gts = batch[4]  # .to(device)
            for i in range(batch_size):
                # set the token id after [EOS] token to 0, to match with the reward calculation
                end_idx = torch.where(greedy_res[i] == EOS)[0]  # position of the first [EOS]
                if len(end_idx) > 0:
                    greedy_res[i, end_idx[0]+1:] = 0
                end_idx = torch.where(gen_result[i] == EOS)[0]  # position of the first [EOS]
                if len(end_idx) > 0:
                    gen_result[i, end_idx[0]+1:] = 0
            reward = get_self_critical_reward(greedy_res, gts, gen_result, EOS=EOS)
            reward = torch.from_numpy(reward).float().to(device)
            loss = rl_crit(sample_logprobs, gen_result.data, reward)

            # parameters that were not used in producing loss
            text_pred_scores, img_pred_scores = pred_scores
            loss += 0 * torch.sum(img_pred_scores)
            loss += 0 * torch.sum(text_pred_scores)

        else:
            if iteration_t2i:  # image generation
                token_ids = input_token_ids  # t2i: do not mask text
                visual_mask = _visual_mask
            else:  # caption generation
                token_ids = masked_token_ids  # i2t: mask text
                visual_mask = None

            pred_scores = generator(grid_features, token_ids, attention_mask, visual_mask, cluster_index)
            text_pred_scores, img_pred_scores = pred_scores  # [bsz,L,30522], [bsz,L,1601]

            if iteration_t2i:
                # involve in loss but ignore it by multiplying 0
                loss += 0 * torch.sum(text_pred_scores)
                img_pred_scores = img_pred_scores[:, :visual_token_num, :]
                img_dim = img_pred_scores.size(-1)
                img_pred_scores = img_pred_scores.contiguous().view(-1, img_dim)
                img_pred_scores = img_pred_scores[mask_position]

                tmp_cluster_index = batch[0].to(device)
                tgt_mgcluster_token_ids = tmp_cluster_index.view(-1)[mask_position]
                loss += mgcluster_criterion(img_pred_scores, tgt_mgcluster_token_ids)

            else:
                mask_position = (masked_token_ids == MASK).to(torch.long).view(-1)
                mask_position = mask_position.nonzero().squeeze()
                if len(masked_token_ids) == 1:  # bsz == 1
                    mask_position = mask_position.unsqueeze(0)
                gt_token_ids = input_token_ids.view(-1)[mask_position]

                loss += 0 * torch.sum(img_pred_scores)

                text_pred_scores = text_pred_scores[:, visual_token_num:, :]
                text_dim = text_pred_scores.size(-1)
                text_pred_scores = text_pred_scores.contiguous().view(-1, text_dim)
                text_pred_scores = text_pred_scores[mask_position]

                mle_loss = criterion(text_pred_scores, gt_token_ids)
                loss += mle_loss

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        if config.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (iteration + 1) % config.gradient_accumulation_steps == 0:
            if config.fp16:
                clip_grad_norm_(amp.master_params(optimizer), config.solver.grad_clip)
            else:
                clip_grad_norm_(generator.parameters(), config.solver.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_time = time.time() - end
        end = time.time()

        if iteration % log_time == 0 or iteration == max_iter or iteration % log_time == 1:
            isnan = False
            if torch.isnan(loss):
                logger.info('loss is nan')
                isnan = True

            logger.info(
                '  '.join([
                    "iter: {iter}", "time: {time:.2f}",
                    "mem: {mem:.1f}",
                    "lr: {lr:.6f}", "loss: {loss:.2f}",
                ]).format(
                    iter=iteration, loss=loss,
                    lr=optimizer.param_groups[0]["lr"],
                    time=batch_time, mem=torch.cuda.max_memory_allocated() / 1024.0 ** 3,
                ))

            if task == 'rl_i2t' and reward.mean() != 0:
                mean_reward = reward.mean()
                scst_reward.append(mean_reward.item())
                logger.info('  '.join(["scst_reward: {scst_reward:.2f}", ]).format(scst_reward=np.mean(scst_reward), ))

            if config.mse_loss:
                logger.info('  '.join(["image_mse_loss: {image_mse_loss:.2f}"]).format(image_mse_loss=image_mse_loss))

            if config.use_clip_score and clip_score.mean() != 0:
                logger.info('  '.join(["clip_score_mean: {clip_score_mean:.2f}"])
                            .format(clip_score_mean=torch.mean(clip_score).item()))

            if isnan:
                exit()

        if iteration % checkpoint_time == 0 or iteration == max_iter:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if (iteration % config.test_interval == 0 or iteration == max_iter) and get_rank() == 0:
            # testing is performed with one-gpu, don't test for the non-master process
            model_name = format(iteration, '07d')
            model_path = os.path.join(save_dir, "model_{:07d}.pth".format(iteration))

            logger.info(f"Start image captioning testing")
            test(generator, device, model_path, save_dir, model_name, logger, tokenizer=tokenizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config.merge_from_list(args.opts)
    config.freeze()

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    if config.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group("nccl", init_method="env://")
        synchronize()

    id = config.exp_id if config.exp_id != '' else 'train'
    save_dir = os.path.join(config.save_dir, id)
    mkdir(save_dir)
    logger = setup_logger("train", save_dir, get_rank())
    logger.info("Running with config:\n{}".format(config))

    tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)

    arguments = {'iteration': 0}

    bert_config = LxmertConfig(type_vocab_size=2, vocab_size=num_tokens)

    bert_config.myconfig = config
    generator = Generator(bert_config)
    device = torch.device(config.device)
    generator = generator.to(device)

    if config.use_clip_score:
        from loss import ClipScorer
        clip_scorer = ClipScorer(device=device)

    optimizer = AdamW(
        params=generator.parameters(),
        lr=config.solver.lr,
        weight_decay=config.solver.weight_decay,
        betas=config.solver.betas,
        eps=config.solver.eps
    )

    if config.scheduler.method == 'ConstantLRSchedule':
        scheduler = get_constant_schedule(optimizer=optimizer)
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.scheduler.warmup_steps,
            num_training_steps=config.scheduler.max_steps
        )

    amp_state = None
    if config.fp16:
        from apex import amp
        loss_scale = 'dynamic' if config.loss_scale < 0 else config.loss_scale
        # https://nvidia.github.io/apex/amp.html
        # loss_scale: If loss_scale is a float value, use this value as the static (fixed) loss scale.
        # If loss_scale is the string "dynamic", adaptively adjust the loss scale over time.
        # Dynamic loss scale adjustments are performed by Amp automatically.
        generator, optimizer = amp.initialize(generator, optimizer, opt_level=config.opt_level, loss_scale=loss_scale)
        amp_state = amp.state_dict()

    checkpointer = Checkpointer(
        model=generator,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        save_to_disk=get_rank() == 0,
        logger=logger,
        amp=amp_state
    )

    latest_model_path = ''
    if config.resume_from_latest:  # resume from the latest model of this exp_id
        save_exp_path = os.path.join(config.save_dir, config.exp_id)
        model_paths = [os.path.join(save_exp_path, f) for f in os.listdir(save_exp_path)
                       if os.path.isfile(os.path.join(save_exp_path, f)) and ('.pth' in f)]
        latest_iteration = 0
        for model_path in model_paths:
            iteration = re.search('model_(.*).pth', model_path).group(1)
            if int(iteration) > latest_iteration:
                latest_iteration = int(iteration)
                latest_model_path = model_path
        logger.info("Assume latest model path {}".format(latest_model_path))

    if config.resume_from_latest and os.path.isfile(latest_model_path):
        load_model_path = latest_model_path
        logger.info("Resume from the latest model {}".format(load_model_path))
    elif os.path.isfile(config.model_path):
        load_model_path = config.model_path
        logger.info("Resume from the specific model {}".format(load_model_path))
    else:
        logger.info("!!!!!!!!!!!!!!!!! Not resume from the latest or a specific model. !!!!!!!!!!!!!!!!!")
        load_model_path=None

    if load_model_path is not None and os.path.isfile(load_model_path):
        logger.info("Loading pretrained model: {}".format(load_model_path))
        extra_checkpoint_data = checkpointer.load(load_model_path, config.load_model_only)
        if not config.restart_iteration:
            # arguments: {'iteration': 0} -> {'iteration': iteration_of_load_model}
            arguments.update(extra_checkpoint_data)
    elif os.path.exists(config.pretrained_model):
        logger.info("Loading weights from pretrained model: {}".format(config.pretrained_model))
        generator.load_weights(config.pretrained_model)
    else:
        logger.info("Not loading weights from pretrained bert or other models.")

    if config.use_clip_score:
        from utils.gan import GAN
        generator.GAN = GAN(device=torch.device(config.device), g_ckpt_path=config.pretrained_generator)
        print("Init GAN.")

    generator.to(device)

    dataset = COCOCaptionDataset(
        root_anno=config.data_dir_anno,
        root_feat=config.data_dir_feat,
        split=config.train_split,
        config=config,
        tokenizer=tokenizer,
    )

    data_loader = make_data_loader(
        dataset=dataset,
        collate_fn=collate_fn_train,
        batch_size=config.samples_per_gpu,
        num_workers=config.num_workers,
        max_iter=config.scheduler.max_steps,
        split=config.train_split,
        is_distributed=config.distributed,
        start_iter=arguments['iteration'],
    )

    if config.distributed:
        generator = DistributedDataParallel(
            module=generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    else:
        generator = torch.nn.DataParallel(generator)
        if config.use_clip_score:
            clip_scorer.model = torch.nn.DataParallel(clip_scorer.model)

    train(generator=generator,
          optimizer=optimizer,
          data_loader=data_loader,
          scheduler=scheduler,
          checkpointer=checkpointer,
          device=device,
          arguments=arguments,
          save_dir=save_dir,
          config=config,
          clip_scorer=clip_scorer if config.use_clip_score else None
          )
