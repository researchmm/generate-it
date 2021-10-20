import argparse
import json
import logging
import os
import re
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers.models.lxmert.modeling_lxmert import LxmertConfig
from modeling import Generator

from config import _C as config
from utils.tokenizer import get_tokenizer
from dataset import COCOCaptionDataset, collate_fn_test
from utils import mkdir
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger

from evaluate_captions import eval_captions

def inference(generator, data_loader, device, tokenizer=None):
    EOS = tokenizer.convert_tokens_to_ids('.')
    MASK = tokenizer.mask_token_id
    PAD = tokenizer.pad_token_id
    SEP = tokenizer.sep_token_id

    logger = logging.getLogger("inference")
    logger.info("Start inferencing")
    generator.eval()

    img2cap_pred = dict()

    for iteration, batch in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
        B = batch[1].size(0)

        if config.use_grid_cluster and not config.use_mix_feature:
            cluster_index = batch[2].to(device)
        else:
            cluster_index = None

        if config.use_grid_feat or config.use_mix_feature:
            grid_features = batch[0].to(device)  # (N, 64, 2048), float
        else:
            grid_features = None

        pred_list = list()

        with torch.no_grad():
            text_token_num = config.text_length + 1  # +1 for [SEP]

            position_id = torch.arange(0, text_token_num, dtype=torch.long, device=device)
            position_ids = position_id.unsqueeze(0).repeat(B, 1)

            masked_token_ids = position_ids.new_full((B, text_token_num), MASK, dtype=torch.long)
            masked_token_ids[:, -1] = SEP  # important to be consistent with training process

            visual_token_num = config.grid_size * config.grid_size

            attention_mask = position_ids.new_zeros((B, text_token_num))
            _attention_mask = attention_mask.new_ones((B, visual_token_num))
            attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)

            # [CLS] in the first token, to be consistent with X-LXMERT
            masked_token_ids[:, 0] = tokenizer.cls_token_id
            attention_mask[:, visual_token_num] = 1  # attend to [CLS] token

            for step in range(1, text_token_num):
                attention_mask[:, visual_token_num + step] = 1  # attend to tokens before current step
                pred_scores = generator(grid_features, masked_token_ids, attention_mask,
                                        visual_mask=None, cluster_index=cluster_index)

                pred_scores = pred_scores[0]

                pred_probs = F.softmax(pred_scores[:, visual_token_num + step, :], dim=-1)
                pred_token_probs, pred_token_ids = pred_probs.max(dim=-1)

                masked_token_ids[:, step] = pred_token_ids
                if pred_token_ids.sum() == (EOS * len(pred_token_ids)):  # all captions are ending
                    break

            pred_list.append(masked_token_ids[:, :config.text_length].cpu().numpy())  # (N, L)

        image_ids = list(batch[1].cpu().numpy())
        for batch_id, image_id in enumerate(image_ids):
            cap_pred = tokenizer.decode(pred_list[0][batch_id], end_flags=[EOS])
            cap_pred = re.sub(r'\b(\w+)( \1\b)+', r'\1', cap_pred)
            img2cap_pred[str(image_id)] = [{'caption': cap_pred}]

    return img2cap_pred

def save_pred(img2cap_pred, save_dir, model_name, logger):
    img2cap_pred_path = os.path.join(save_dir, 'caption_results_' + model_name)
    img2cap_pred_path += '_len' + str(config.text_length)
    img2cap_pred_path = img2cap_pred_path + '.json'

    logger.info(f"Saving results to {img2cap_pred_path}")
    with open(img2cap_pred_path, 'w') as f:
        json.dump(img2cap_pred, f)

    return img2cap_pred_path


def test(generator, device, model_path='', save_dir='', model_name='', logger=None, tokenizer=None):
    if os.path.exists(model_path):
        g_checkpointer = Checkpointer(model=generator, logger=logger)
        g_checkpointer.load(model_path, True)

    dataset = COCOCaptionDataset(
        root_anno=config.data_dir_anno,
        root_feat=config.data_dir_feat,
        split='test',
        config = config,
        tokenizer=tokenizer
    )
    data_loader = make_data_loader(
        dataset=dataset,
        collate_fn=collate_fn_test,
        batch_size=config.samples_per_gpu,
        num_workers=config.num_workers,
        split='test'
    )

    img2cap_pred = inference(generator, data_loader, device, tokenizer=tokenizer)

    img2cap_pred_path = save_pred(img2cap_pred, save_dir, model_name, logger)

    eval_captions(img2cap_pred_path, gt_caption=config.id2captions_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config.merge_from_list(args.opts)
    config.freeze()

    if config.xlxmert_zeroshot:
        model_name = 'xlxmert_zeroshot'
    else:
        model_name = re.search('(.*).pth', config.model_path.split('/')[-1]).group(1)
    save_dir = os.path.join(config.save_dir, config.exp_id)
    mkdir(save_dir)
    logger = setup_logger("inference", save_dir, 0, filename='infer_' + model_name + '.log')
    logger.info("Running with config:\n{}".format(config))

    tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)

    device = torch.device(config.device)

    bert_config = LxmertConfig(type_vocab_size=2, vocab_size=num_tokens)

    bert_config.myconfig = config
    generator = Generator(bert_config)

    generator = generator.to(device)

    test(generator, device, config.model_path, save_dir, model_name, logger, tokenizer=tokenizer)
