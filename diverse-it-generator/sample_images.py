from config import _C as config
from modeling import Generator
import argparse
import re
import torch
from tqdm import tqdm
import json
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from utils.logger import setup_logger
from utils.tokenizer import get_tokenizer
from utils.checkpointer import Checkpointer

from transformers.modeling_bert import BertConfig

def clean_text(sent):
    sent = sent.replace("\ufffd\ufffd", " ")
    sent = sent.replace("\n", ' ')
    sent = sent.replace(" .", '.')
    sent = sent.replace("/", ' ')
    sent = sent.replace("  ", ' ')
    sent = " ".join(sent.split())
    return sent

def get_captions(caption_path='assets/captions_5sample.txt', karpathy_test=''):
    image_ids = []
    if karpathy_test == '' and os.path.exists(caption_path):
        with open(caption_path) as f:
            captions = f.readlines()
    else:
        idx2imgid = {}
        with open(config.karpathy_test_result_example, 'r') as fr:
            example = json.load(fr)
            for i in range(len(example)):
                idx2imgid[i] = str(example[i]['image_id'])

        with open(karpathy_test, 'r') as fr:
            test_gts_dict = json.load(fr)
        captions = []
        for i in range(len(test_gts_dict)):
            # idx = 0  # random.randint(0, 4)
            for idx in range(5):
                captions.append(test_gts_dict[idx2imgid[i]][idx])
            image_ids.append(idx2imgid[i])

    captions = [clean_text(sent) for sent in captions]
    if config.num_sample_captions == 1:
        captions = captions[::5]  # sample a caption for each image
    return captions, image_ids


def sample_image(sentences, n_steps=None, generator=None, tokenizer=None, config=None, autoregressive=False,
                 which_caption=1, save_dir='img_samples', start_idx=0, gen_pred_img=False):
    generator.eval()

    EOS = tokenizer.convert_tokens_to_ids('.')
    SEP = tokenizer.sep_token_id

    input_token_ids = []
    input_token_id = []
    for i, caption in enumerate(sentences):
        if i % config.num_sample_captions == 0:
            input_token_id = []
        caption = tokenizer.encode(caption)
        caption = caption[:config.text_length]
        offset = config.text_length - len(caption)
        input_token_id += caption + [EOS] * offset + [SEP]
        if (i + 1) % config.num_sample_captions == 0:
            input_token_ids.append(input_token_id)
    B = len(input_token_ids)
    input_token_ids = torch.tensor(input_token_ids, dtype=torch.long).cuda()

    total_length = 0
    for i in range(config.num_sample_captions):
        position_id = torch.arange(0, config.text_length + 1, dtype=torch.long, device=device)
        position_id = position_id.unsqueeze(0).repeat(B, 1)

        total_length += (config.text_length + 1)

        if i == 0:
            segment_type_ids = position_id.new_full((B, config.text_length + 1), i + 1, dtype=torch.long)
            position_ids = position_id
        else:
            segment_type_ids = torch.cat(
                (segment_type_ids, position_id.new_full((B, config.text_length + 1), i + 1, dtype=torch.long)), dim=1)
            position_ids = torch.cat((position_ids, position_id), dim=1)

    attention_mask = (input_token_ids != PAD).float()  # attend to all gts

    visual_token_num = config.grid_size * config.grid_size

    visual_token_type = position_ids.new_full((position_ids.size(0), visual_token_num), config.num_sample_captions + 1)
    token_type_ids = torch.cat((visual_token_type, segment_type_ids), dim=1)

    _attention_mask = attention_mask.new_ones((B, visual_token_num))
    attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

    visual_mask = torch.ones((B, visual_token_num), dtype=torch.float32, device=device)
    fake_visual_feats = torch.zeros((B, visual_token_num, config.grid_feat_dim), dtype=torch.float32, device=device)
    cluster_index = torch.zeros(B, visual_token_num, dtype=torch.long).cuda()

    if autoregressive:
        n_steps = visual_token_num

    for i in range(n_steps):
        if autoregressive:
            attention_mask[:, i+1:visual_token_num] = 0

        pred_scores = generator(
            grid_features=fake_visual_feats, masked_token_ids=input_token_ids, token_type_ids=token_type_ids,
            position_ids=position_ids, attention_mask=attention_mask,
            visual_mask=visual_mask, cluster_index=cluster_index)

        # [bsz,L,30522], [bsz,L,1601]
        # text_pred_scores, img_pred_scores = pred_scores
        pred_code_logit = pred_scores[1][:, :visual_token_num, :]

        pred_code_prob = torch.softmax(pred_code_logit, dim=2)
        pred_prob, pred_cluster_index = pred_code_prob.max(dim=2)

        # the generator() will generate pred_scores->pred_code_prob->pred_cluster_index at all positions
        # while we only want to update the index of positions indicated by 1 in visual_mask (lowest_prob positions)
        # other positions have high probs and are fixed
        cluster_index = torch.where(visual_mask.bool(), pred_cluster_index, cluster_index)

        fake_visual_feats = generator.cluster_codebook(cluster_index)

        if autoregressive:
            visual_mask[:, i] = 0
        else:
            # Linear decay for mask updates (Mask-Predict)
            # 'i+1': the visual_mask is updated after generator(), so the first ratio (1) has been used
            ratio = (n_steps - (i+1)) / n_steps
            n_mask = int(ratio * visual_token_num)
            visual_mask = torch.zeros((B, visual_token_num), dtype=torch.float32, device=device)
            lowest_prob, lowest_arg = pred_prob.topk(n_mask, dim=1, largest=False)
            visual_mask.scatter_(1, lowest_arg, 1)

    if gen_pred_img:
        # which_caption ranges from [1,5], while the sentences index from 0, so 'which_caption-1'
        captions_name = sentences[which_caption-1::config.num_sample_captions]
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

    return


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
    parser.add_argument("--demo_captions", type=bool, default=False)
    args = parser.parse_args()

    config.merge_from_list(args.opts)
    model_name = re.search('(.*).pth', config.model_path.split('/')[-1]).group(1)
    save_dir = os.path.join(config.save_dir, config.exp_id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = setup_logger("inference", save_dir, 0, filename='infer_' + model_name + '.log')
    logger.info("Running with config:\n{}".format(config))

    tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)

    device = torch.device(config.device)

    num_types = config.num_sample_captions + 2  # 2 is for visual type and [PAD] type
    bert_config = BertConfig(type_vocab_size=num_types, num_hidden_layers=config.num_hidden_layers,
                             vocab_size_or_config_json_file=num_tokens)
    bert_config.myconfig = config
    generator = Generator(bert_config)
    generator = generator.to(device)

    load_model_path, load_iteration = get_model_path(logger, config)

    g_checkpointer = Checkpointer(model=generator, logger=logger)
    g_checkpointer.load(load_model_path, True)

    if config.demo_captions:
        captions, image_ids = get_captions(caption_path='assets/captions_5sample.txt')
    else:
        captions, image_ids = get_captions(karpathy_test=config.test_gts_dict)

    input_size = len(captions)
    batch_size = config.samples_per_gpu
    assert config.samples_per_gpu % config.num_sample_captions == 0, \
        "The config.samples_per_gpu should be a multiple of config.num_sample_captions"

    if batch_size > input_size:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = input_size

    img_save_dir = config.img_save_dir + '/' + config.exp_id + load_iteration
    logger.info("Images will be saved at {}".format(img_save_dir))
    j = 1
    for i in tqdm(range((input_size // batch_size) + 1)):  # +1 for the last batch (maybe smaller than batch_size)
        start = i * batch_size
        end = start + batch_size
        batch_captions = captions[start:end]
        img_batch_size = int(len(batch_captions)/config.num_sample_captions)
        if len(batch_captions) == 0:
            break

        sample_image(batch_captions, n_steps=4, generator=generator, tokenizer=tokenizer,
                     config=config, autoregressive=False,
                     which_caption=1, save_dir=img_save_dir, start_idx=j, gen_pred_img=True)
        j+= img_batch_size

    if not config.demo_captions:
        results = {}
        results['img_save_dir'] = img_save_dir
        results['split'] = 'karpathy_test'

        from evaluation.pytorch_fid import fid_score

        fid_paths = [config.gt_img_path, img_save_dir]
        print('Evaluating FID: ', fid_paths)
        fid_value = fid_score.calculate_fid_given_paths(paths=fid_paths)
        print('FID: ', fid_value, fid_paths)
        results['FID'] = str(fid_value)
        result_path = img_save_dir + '_F' + str(round(fid_value, 2)) + '.json'
        print("Saving results to path: ", result_path)
        print("Results:", results)
        with open(result_path, 'w') as fw:
            json.dump(results, fw)