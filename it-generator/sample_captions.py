from config import _C as config
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import h5py
import numpy as np
import re
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from utils.logger import setup_logger
from utils.tokenizer import get_tokenizer
from utils.checkpointer import Checkpointer

from transformers.models.lxmert.modeling_lxmert import LxmertConfig
from modeling import Generator
from evaluate_captions import eval_captions

def clean_text(sent):
    sent = sent.replace("\ufffd\ufffd", " ")
    sent = sent.replace("\n", ' ')
    sent = sent.replace(" .", '.')
    sent = sent.replace("/", ' ')
    sent = sent.replace("  ", ' ')
    sent = " ".join(sent.split())
    return sent

def get_captions(caption_path='captions_1sample.txt', karpathy_test='', test_val2014=False):
    image_ids = []
    filenames = []
    if test_val2014:
        with open(config.dataset_coco) as f:
            all_captions = json.load(f)
            all_captions = all_captions['images']
        captions = []
        # filter out karpathy test split firstly
        for item in all_captions:
            if item['split'] == 'test':
                # image_id = item['filename'].split('.')[0]  # e.g., 'COCO_train2014_000000475546'
                image_id = str(item['cocoid'])  # e.g., '475546'
                image_ids.append(image_id)
                filenames.append(item['filename'])
                for sent_idx, c in enumerate(item['sentences']):
                    caption = ' '.join(c['tokens']) + '.'
                    captions.append(caption)
                    if sent_idx == 4:
                        break
        for item in all_captions:
            if item['split'] == 'val':
                # image_id = item['filename'].split('.')[0]  # e.g., 'COCO_train2014_000000475546'
                image_id = str(item['cocoid'])  # e.g., '475546'
                image_ids.append(image_id)
                filenames.append(item['filename'])
                for sent_idx, c in enumerate(item['sentences']):
                    caption = ' '.join(c['tokens']) + '.'
                    captions.append(caption)
                    if sent_idx == 4:
                        break
        for item in all_captions:
            if item['split'] == 'restval':
                # image_id = item['filename'].split('.')[0]  # e.g., 'COCO_train2014_000000475546'
                image_id = str(item['cocoid'])  # e.g., '475546'
                image_ids.append(image_id)
                filenames.append(item['filename'])
                for sent_idx, c in enumerate(item['sentences']):
                    caption = ' '.join(c['tokens']) + '.'
                    captions.append(caption)
                    if sent_idx == 4:
                        break
                if len(image_ids) == 30000:
                    break
    elif karpathy_test == '' and os.path.exists(caption_path):
        with open(caption_path) as f:
            captions = f.readlines()
    else:  # evaluate karpathy test split
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
                captions.append(test_gts_dict[idx2imgid[i]][idx])  # sample a caption for each image
            image_ids.append(idx2imgid[i])

    captions = [clean_text(sent) for sent in captions]
    captions = captions[::5]  # sample a caption for each image, assume each image has five gt captions

    return captions, image_ids, filenames


def sample_caption(visual_features=None, generator=None, tokenizer=None,
                     config=None, cluster_index=None, sample_mode='greedy',
                     masked_token_ids=None, attention_mask=None, return_res_prob=False):
    device = torch.device(config.device)

    B = len(visual_features) if visual_features is not None else len(cluster_index)
    EOS = tokenizer.convert_tokens_to_ids('.')
    MASK = tokenizer.mask_token_id
    SEP = tokenizer.sep_token_id
    visual_token_num = config.grid_size * config.grid_size
    masked_token_ids[:] = MASK
    masked_token_ids[:, 0] = tokenizer.cls_token_id  # [CLS] in the first token, to be consistent with X-LXMERT
    attention_mask[:, visual_token_num:] = 0
    attention_mask[:, visual_token_num] = 1  # attend to [CLS] token

    text_token_num = config.text_length+1
    update_mask = torch.zeros((B, text_token_num), dtype=torch.float32, device=device)

    if return_res_prob:
        res_prob = torch.zeros((B, text_token_num), dtype=torch.float32, device=device)
    else:
        res_prob = None

    pred_scores = None
    for i in range(1, config.text_length):
        attention_mask[:, i+visual_token_num] = 1
        update_mask[:, :] = 0  # necessary
        update_mask[:, i] = 1

        masked_token_ids[:, -1] = SEP

        pred_scores = generator(grid_features=visual_features, masked_token_ids=masked_token_ids.clone(),
                                attention_mask=attention_mask, visual_mask=None, cluster_index=cluster_index)

        text_pred_scores, _ = pred_scores
        pred_code_logit = text_pred_scores[:, visual_token_num:visual_token_num+text_token_num, :]

        batch_size, length, vocab_size = pred_code_logit.shape
        pred_code_logit = pred_code_logit.reshape(-1, vocab_size)

        pred_code_prob = torch.softmax(pred_code_logit, dim=-1).detach()
        if sample_mode == 'greedy':
            pred_prob, pred_index = pred_code_prob.max(dim=-1)
        elif sample_mode == 'sample':
            pred_index = torch.multinomial(pred_code_prob, num_samples=1, replacement=True)
            pred_prob = torch.gather(F.log_softmax(pred_code_logit.squeeze_(1), dim=-1), 1, pred_index)
        else:
            raise NotImplementedError

        pred_prob = pred_prob.reshape(batch_size, length)
        pred_index = pred_index.reshape(batch_size, length)

        if return_res_prob:
            res_prob = torch.where(update_mask.bool(), pred_prob, res_prob)
        masked_token_ids[:, :text_token_num] = \
            torch.where(update_mask.bool(), pred_index, masked_token_ids[:, :text_token_num])

        if pred_index[:, i].sum() == (EOS * B):  # all captions are ending
            break

    # [tokenizer.decode(res.cpu().numpy(), end_flags=[EOS]) for res in res_index]
    res_index = masked_token_ids[:, :text_token_num].clone()
    res_index[:, -1] = EOS  # SEP->EOS to ending the caption
    return res_index, res_prob, pred_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--gt_caption", type=str, default='id2captions_test.json')
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

    bert_config = LxmertConfig(type_vocab_size=2, vocab_size=num_tokens)
    bert_config.myconfig = config
    generator = Generator(bert_config)
    generator = generator.to(device)

    g_checkpointer = Checkpointer(model=generator, logger=logger)
    g_checkpointer.load(config.model_path, True)

    captions, image_ids, filenames = get_captions(karpathy_test=config.test_gts_dict)

    input_size = len(captions)
    batch_size = config.samples_per_gpu

    if batch_size > input_size:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = input_size

    if config.use_grid_feat:
        grid_feat_valid = h5py.File(config.grid_feat_path_valid, 'r')
    else:
        grid_feat_valid = None

    image2gen_sentences = {}
    for i in tqdm(range((input_size // batch_size) + 1)):  # +1 for the last batch (maybe smaller than batch_size)
        start = i * batch_size
        end = start + batch_size
        batch_captions = captions[start:end]
        img_batch_size = int(len(batch_captions))
        if len(batch_captions) == 0:
            break

        if config.use_grid_feat:
            visual_features = []
            for k in range(0, min(img_batch_size, len(image_ids))):
                img_name = 'COCO_val2014_' + str(image_ids[k]).zfill(12)
                grid_feature = grid_feat_valid[f'{img_name}/features'][:]
                grid_feature = np.reshape(grid_feature, (config.grid_size ** 2, config.grid_feat_dim))
                grid_feature = torch.from_numpy(grid_feature)
                visual_features.append(grid_feature)
            visual_features = torch.stack(visual_features, dim=0).cuda()
        else:
            visual_features = None

        generator.eval()
        gen_captions, _, _ = sample_caption(visual_features=visual_features, generator=generator,
                                            tokenizer=tokenizer, config=config, sample_mode='greedy')

        gen_captions = gen_captions.cpu().numpy()
        for k in range(len(gen_captions)):
            gen_sentence = tokenizer.decode(gen_captions[k], end_flags=[EOS])
            cap_pred = re.sub(r'\b(\w+)( \1\b)+', r'\1', gen_sentence)
            image2gen_sentences[image_ids[k]] = [{'caption': gen_sentence}]

    img2cap_pred_path = 'img2cap_pred.json'
    with open(img2cap_pred_path, 'w') as fw:
        json.dump(image2gen_sentences, fw)
    eval_captions(img2cap_pred_path, gt_caption=config.gt_caption)
