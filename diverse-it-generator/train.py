import argparse
import logging
import os
import time
import random
import os.path, sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from transformers.modeling_bert import BertConfig
from transformers.optimization import AdamW, WarmupCosineSchedule

from config import _C as config
from dataset import COCOCaptionDataset, collate_fn_train
from modeling import Generator, LabelSmoothingLoss
from modeling import UnlikelihoodLabelSmoothingLoss
from utils import get_rank, mkdir, synchronize
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger
from inference import test
from utils.tokenizer import get_tokenizer
import re


def train(generator, optimizer, data_loader, scheduler, checkpointer,
          device, arguments, save_dir, config):
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
    mgcluster_criterion = LabelSmoothingLoss(config.grid_cluster_num)

    end = time.time()
    optimizer.zero_grad()

    use_discrete = config.use_grid_cluster and not config.use_grid_feat

    for iteration, batch in enumerate(data_loader, start_iter):

        iteration = iteration + 1
        arguments['iteration'] = iteration

        task = config.iteration_tasks[iteration%len(config.iteration_tasks)]
        iteration_t2i = 1 if 't2i' in task else 0

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
        segment_type_ids = batch[4].to(device)  # (N, L), long
        mask_idxs = batch[5].to(device)  # (N, L), long
        # arange from 0 to the total sequence length of the concatenated five captions
        position_ids = batch[6].to(device)  # (N, L), long

        batch_size = input_token_ids.size(0)
        visual_token_num = config.grid_size * config.grid_size

        if iteration_t2i:  # attend to img
            # for captions
            attention_mask = (masked_token_ids != PAD).float()  # attend to all gts

            # for images
            # attend to all visual tokens for non-autoregressive models, but mask some visual tokens for prediction
            num_masks = random.randint(max(1, int(0.1 * visual_token_num)), visual_token_num)
            selected_idx = random.sample(range(visual_token_num), num_masks)
            _visual_mask = torch.zeros((batch_size, visual_token_num), dtype=torch.float32, device=device)
            _visual_mask[:, selected_idx] = 1
            mask_position = (_visual_mask == 1).to(torch.long).view(-1)
            mask_position = mask_position.nonzero().squeeze()
            token_ids = input_token_ids  # t2i: do not mask text
            visual_mask = _visual_mask

        else:  # mask text
            attention_mask = (masked_token_ids != PAD).float()
            attention_mask[segment_type_ids > mask_idxs.unsqueeze(1)] = 0
            # only mask the target caption, do not attend on other four captions
            # do not mask ([MASK] token) the target caption for generation
            attention_mask[segment_type_ids == mask_idxs.unsqueeze(1)] = 1
            if config.autoregressive == 1:
                for i in range(len(attention_mask)):
                    mask_idx = torch.where(masked_token_ids[i] == MASK)[0][0]  # position of [MASK]
                    # the end idx of this caption (including all [eos])
                    end_idx = torch.where(attention_mask[i] == 1)[0][-1]
                    attention_mask[i, mask_idx + 1:end_idx + 1] = 0  # do not attend to tokens after [MASK]

            token_ids = masked_token_ids  # i2t: mask text
            visual_mask = None

        _attention_mask = torch.ones((batch_size, visual_token_num), dtype=torch.float32, device=device)
        attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

        visual_token_type = position_ids.new_full((position_ids.size(0), visual_token_num),
                                              config.num_sample_captions + 1)
        token_type_ids = torch.cat((visual_token_type, segment_type_ids), dim=1)

        pred_scores = generator(grid_features, token_ids, token_type_ids, position_ids, attention_mask,
                                visual_mask=visual_mask, cluster_index=cluster_index)

        text_pred_scores, img_pred_scores = pred_scores  # [bsz,L,30522], [bsz,L,1601]

        loss = mle_loss = unlikelihood_loss = 0
        if iteration_t2i:
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

            if config.unlikelihood_training == 1:
                unlikelihood_criterion = UnlikelihoodLabelSmoothingLoss(num_tokens, config=config, tokenizer=tokenizer)
                total_loss, mle_loss, unlikelihood_loss = unlikelihood_criterion(text_pred_scores, gt_token_ids,
                                              input_token_ids, attention_mask[:, visual_token_num:], masked_token_ids)
                loss += total_loss
            else:
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
            logger.info('  '.join(["iter: {iter}", "time: {time:.1f}", "mem: {mem:.1f}",
                                   "lr: {lr:.6f}", "loss: {loss:.2f}",]).format(
                    iter=iteration, time=batch_time,
                    lr=optimizer.param_groups[0]["lr"],
                    mem=torch.cuda.max_memory_allocated() / 1024.0 ** 3, loss=loss))

            if unlikelihood_loss != 0:
                logger.info('  '.join(["unlikelihood_loss: {unlikelihood_loss:.2f}",]).format(
                    unlikelihood_loss=unlikelihood_loss))

            if mle_loss != 0:
                logger.info('  '.join(["mle_loss: {mle_loss:.2f}",]).format(
                    mle_loss=mle_loss))

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

    num_types = config.num_sample_captions + 2  # 2 is for visual type and [PAD] type
    bert_config = BertConfig(type_vocab_size=num_types, num_hidden_layers=config.num_hidden_layers,
                             vocab_size_or_config_json_file=num_tokens)
    bert_config.myconfig = config
    generator = Generator(bert_config)
    device = torch.device(config.device)
    generator = generator.to(device)

    optimizer = AdamW(
        params=generator.parameters(),
        lr=config.solver.lr,
        weight_decay=config.solver.weight_decay,
        betas=config.solver.betas,
        eps=config.solver.eps
    )

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        t_total=config.scheduler.max_steps
    )

    amp_state = None
    if config.fp16:
        from apex import amp
        generator, optimizer = amp.initialize(generator, optimizer, opt_level=config.opt_level,
                                              loss_scale=config.loss_scale)
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
    elif os.path.exists(config.pretrained_bert):
        logger.info("Loading weights from pretrained bert model: {}".format(config.pretrained_bert))
        generator.load_weights(config.pretrained_bert)
    else:
        logger.info("Not loading weights from pretrained bert or other models.")

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

    train(generator=generator,
          optimizer=optimizer,
          data_loader=data_loader,
          scheduler=scheduler,
          checkpointer=checkpointer,
          device=device,
          arguments=arguments,
          save_dir=save_dir,
          config=config,
          )
