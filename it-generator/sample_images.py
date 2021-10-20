from config import _C as config
import argparse
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from utils.logger import setup_logger
from utils.tokenizer import get_tokenizer
from utils.checkpointer import Checkpointer

from transformers.models.lxmert.modeling_lxmert import LxmertConfig
from modeling import Generator

from sample_captions import get_captions


def get_inputs(config, device, B, sentences=None, input_token_ids=None, return_code_logit=False, attention_mask=None):
    visual_token_num = config.grid_size * config.grid_size
    tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)

    if input_token_ids is None and sentences is not None:
        input_token_ids = []
        input_token_id = []
        for i, caption in enumerate(sentences):
            if i % 1 == 0:
                input_token_id = []
            caption = tokenizer.encode(caption)  # has included [CLS] in the beginning and [SEP] in the end
            # X-LXMERT use the following codes to control the max_length of each batch dynamically
            # input_ids = self.tokenizer(
            #             sentences, max_length=max_text_length, truncation=True, return_tensors='pt').input_ids
            caption = caption[:-1]  # firstly remove the last token [SEP]
            caption = caption[:config.text_length]  # then truncate the specific length
            offset = config.text_length - len(caption)
            input_token_id += caption + [SEP] + [PAD] * offset  # finally ensure to add [SEP]
            if (i + 1) % 1 == 0:
                input_token_ids.append(input_token_id)
        input_token_ids = torch.tensor(input_token_ids, dtype=torch.long).cuda()

    if attention_mask is None:
        attention_mask = torch.zeros(input_token_ids.shape, dtype=torch.float32, device=device)
        attention_mask[:, :config.text_length + 1] = 1
        attention_mask[input_token_ids == PAD] = 0  # do not attend to PAD

        _attention_mask = attention_mask.new_ones((B, visual_token_num))
        attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

    visual_mask = torch.ones((B, visual_token_num), dtype=torch.float32, device=device)
    fake_visual_feats = torch.zeros((B, visual_token_num, config.grid_feat_dim), dtype=torch.float32, device=device)
    cluster_index = torch.zeros(B, visual_token_num, dtype=torch.long).cuda()
    if return_code_logit:
        code_logit = torch.zeros(B, visual_token_num, config.grid_cluster_num, dtype=torch.float).cuda()
    else:
        code_logit = None

    return input_token_ids, attention_mask, visual_mask, fake_visual_feats, cluster_index, code_logit


def sample_image(sentences=None, n_steps=4, generator=None, config=None,
                 which_caption=1, save_dir='images',
                 start_idx=0, gen_pred_img=False,
                 input_token_ids=None, attention_mask=None,
                 visual_mask=None, fake_visual_feats=None, cluster_index=None, code_logit=None,
                 sample_mode='greedy', words_embeddings=None, return_code_logit=True):
    B = len(sentences) if sentences is not None else len(input_token_ids)
    visual_token_num = config.grid_size * config.grid_size
    device = torch.device(config.device)
    if fake_visual_feats is None or sentences is None:
        input_token_ids, attention_mask, visual_mask, fake_visual_feats, cluster_index, code_logit =\
            get_inputs(config=config, device=device, B=B, sentences=sentences, input_token_ids=input_token_ids,
                        return_code_logit=return_code_logit, attention_mask=attention_mask)

    pred_scores = None
    for i in range(n_steps):
        pred_scores = generator(grid_features=fake_visual_feats, masked_token_ids=input_token_ids,
                                attention_mask=attention_mask, visual_mask=visual_mask,
                                cluster_index=cluster_index, words_embeddings=words_embeddings)

        _, img_pred_scores = pred_scores
        pred_code_logit = img_pred_scores[:, :visual_token_num, :]
        if return_code_logit:
            code_logit = torch.where(visual_mask.bool().unsqueeze(-1), pred_code_logit, code_logit)

        batch_size, length, vocab_size = pred_code_logit.shape
        pred_code_logit = pred_code_logit.reshape(-1, vocab_size)

        pred_code_prob = torch.softmax(pred_code_logit, dim=-1)  # .detach()
        if sample_mode == 'greedy':
            pred_prob, pred_index = pred_code_prob.max(dim=-1)
        elif sample_mode == 'sample':
            pred_index = torch.multinomial(pred_code_prob, num_samples=1, replacement=True)
            pred_prob = torch.gather(F.log_softmax(pred_code_logit.squeeze_(1), dim=-1), 1, pred_index)
        else:
            raise NotImplementedError

        pred_prob = pred_prob.reshape(batch_size, length)
        pred_index = pred_index.reshape(batch_size, length)

        # the generator() will generate pred_scores->pred_code_prob->pred_index at all positions
        # while we only want to update the index of positions indicated by 1 in visual_mask (lowest_prob positions)
        # other positions have high probs and are fixed
        cluster_index = torch.where(visual_mask.bool(), pred_index, cluster_index)

        # Linear decay for mask updates (Mask-Predict)
        # 'i+1': the visual_mask is updated after generator(), so the first ratio (1) has been used
        ratio = (n_steps - (i + 1)) / n_steps
        n_mask = int(ratio * visual_token_num)
        if n_mask == 0:
            break
        visual_mask = torch.zeros((B, visual_token_num), dtype=torch.float32, device=device)
        # pred_prob = (cluster_prob + pred_prob) / 2
        lowest_prob, lowest_arg = pred_prob.topk(n_mask, dim=1, largest=False)
        visual_mask.scatter_(1, lowest_arg, 1)

    if gen_pred_img:
        # which caption ranges from [1,5] to match with segment_type_ids
        # while the sentences index from 0, so 'which_caption-1'
        if sentences is not None:
            captions_name = sentences[which_caption-1::1]
        else:
            tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)
            captions_name = [tokenizer.decode(res.cpu().numpy(), end_flags=[EOS]) for res in input_token_ids]
        captions_name = [('img' + str(i) + '.' + cap) for i, cap in enumerate(captions_name, start=start_idx)]

        from utils.gan import GAN
        if hasattr(generator, 'module'):
            if not hasattr(generator.module, 'GAN'):
                generator.module.GAN = GAN(device=torch.device(generator.module.myconfig.device),
                                           g_ckpt_path=generator.module.myconfig.pretrained_generator)
            input_visual_feats = generator.module.cluster_codebook(cluster_index) \
                if cluster_index is not None else fake_visual_feats
            generator.module.GAN.generate_image(visual_feats=input_visual_feats,
                                                captions=captions_name, save_dir=save_dir)
        else:
            if not hasattr(generator, 'GAN'):
                generator.GAN = GAN(device=torch.device(generator.myconfig.device),
                                    g_ckpt_path=generator.myconfig.pretrained_generator)
            input_visual_feats = generator.cluster_codebook(cluster_index) \
                if cluster_index is not None else fake_visual_feats
            generator.GAN.generate_image(visual_feats=input_visual_feats,
                                         captions=captions_name, save_dir=save_dir)

    return cluster_index, code_logit, pred_scores


def get_model_path(logger, config):
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
        load_iteration = '_' + re.search('model_(.*).pth', load_model_path).group(1)
    elif os.path.isfile(config.model_path):
        load_model_path = config.model_path
        logger.info("Resume from the specific model {}".format(load_model_path))
        load_iteration = ''
    else:
        load_model_path = ''
        logger.info("!!!!!!!!!!!!!!!!! Not resume from the latest or a specific model. !!!!!!!!!!!!!!!!!")
        load_iteration = ''

    return load_model_path, load_iteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config.merge_from_list(args.opts)

    save_dir = os.path.join(config.save_dir, config.exp_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = setup_logger("inference", save_dir, 0, filename='infer_model.log')
    logger.info("Running with config:\n{}".format(config))

    load_model_path, load_iteration = get_model_path(logger, config)

    img_save_dir = config.img_save_dir + '/' + config.exp_id + load_iteration
    logger.info("Images will be saved at {}".format(img_save_dir))

    tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)

    bert_config = LxmertConfig(type_vocab_size=2, vocab_size=num_tokens)
    bert_config.myconfig = config
    generator = Generator(bert_config)
    device = torch.device(config.device)
    generator = generator.to(device)
    g_checkpointer = Checkpointer(model=generator, logger=logger)
    g_checkpointer.load(load_model_path, True)
    generator.to(device)
    generator.eval()

    captions, image_ids, filenames = get_captions(karpathy_test=config.test_gts_dict, test_val2014=config.test_val2014)

    if config.test_num > 0 and config.test_num < len(captions):
        input_size = config.test_num
    else:
        input_size = len(captions)
    batch_size = config.samples_per_gpu

    if batch_size > input_size:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = input_size

    j = 1
    for i in tqdm(range((input_size // batch_size) + 1)):  # +1 for the last batch (maybe smaller than batch_size)
        start = i * batch_size
        end = start + batch_size
        batch_captions = captions[start:end]
        img_batch_size = len(batch_captions)
        if img_batch_size == 0:
            break

        sample_image(sentences=batch_captions, generator=generator, config=config, which_caption=1,
                     save_dir=img_save_dir, start_idx=j, gen_pred_img=True, sample_mode='greedy')

        j+= img_batch_size
