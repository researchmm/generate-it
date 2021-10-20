import argparse
import json
import logging
import os
import re
import time
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.modeling_bert import BertConfig

from config import _C as config
from utils.tokenizer import get_tokenizer
from dataset import COCOCaptionDataset, collate_fn_test
from modeling import Generator
from utils import mkdir
from utils.checkpointer import Checkpointer
from utils.dataloader import make_data_loader
from utils.logger import setup_logger

from utils.generation import custom_generate
from evaluate import eval_div_acc

def inference(generator, data_loader, device, tokenizer=None):
    EOS = tokenizer.convert_tokens_to_ids('.')
    MASK = tokenizer.mask_token_id
    PAD = tokenizer.pad_token_id
    SEP = tokenizer.sep_token_id

    logger = logging.getLogger("inference")
    logger.info("Start inferencing")
    generator.eval()

    img2cap_pred = dict()

    end = time.time()
    for iteration, batch in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
        iteration = iteration + 1
        B = batch[1].size(0)

        if (not config.use_mix_feature) and config.use_grid_cluster:
            cluster_index = batch[2].to(device)
            grid_features = None
        else:
            grid_features = batch[0].to(device)  # (N, 64, 2048), float
            cluster_index = None

        pred_list = list()

        with torch.no_grad():
            batch_id = torch.arange(0, B, 1, device=device).unsqueeze(1)

            total_length = 0
            for i in range(config.num_sample_captions):
                high = config.text_length

                position_id = torch.arange(0, high + 1, dtype=torch.long, device=device)
                position_id = position_id.unsqueeze(0).repeat(B, 1)

                total_length += (high + 1)

                if i == 0:
                    segment_type_ids = position_id.new_full((B, high + 1), i + 1, dtype=torch.long)
                    masked_token_ids = segment_type_ids.new_full((B, high + 1), MASK)
                    position_ids = position_id

                else:
                    segment_type_ids = torch.cat(
                        (segment_type_ids, position_id.new_full((B, high + 1), i + 1, dtype=torch.long)), dim=1)
                    masked_token_ids = torch.cat(
                        (masked_token_ids, segment_type_ids.new_full((B, high + 1), MASK)), dim=1)
                    position_ids = torch.cat((position_ids, position_id), dim=1)
                masked_token_ids[:, -1] = SEP  # important to be consistent with training process

            attention_mask = position_ids.new_zeros((B, total_length))

            visual_token_num = config.grid_size * config.grid_size
            lang_start = visual_token_num

            visual_token_type = position_ids.new_full((position_ids.size(0), visual_token_num),
                                                  config.num_sample_captions + 1)
            token_type_ids = torch.cat((visual_token_type, segment_type_ids), dim=1)

            _attention_mask = attention_mask.new_ones((B, visual_token_num))
            attention_mask = torch.cat((_attention_mask, attention_mask), dim=1)


            if config.autoregressive == 1:
                cur_start = 0
                for i in range(config.num_sample_captions):
                    high = config.text_length

                    attention_mask[:, lang_start: lang_start + cur_start] = 1  # condition on generated captions

                    if config.num_beams > 1:
                        masked_token_ids = custom_generate(config, None, None,
                                            generator, high, lang_start, cur_start,
                                            grid_features, masked_token_ids, token_type_ids,
                                            position_ids, attention_mask,
                                            eos_token_id=EOS, pad_token_id=PAD, cluster_index=cluster_index)
                    else:
                        for step in range(high):
                            attention_mask[:, lang_start + cur_start + step] = 1  # attend to tokens before current step
                            pred_scores = generator(
                                grid_features, masked_token_ids, token_type_ids,
                                position_ids, attention_mask,
                                visual_mask=None, cluster_index=cluster_index)

                            pred_scores = pred_scores[0]

                            if config.custom_generation:
                                if config.consider_generated_sentence:  # more diverse
                                    # input all previous generated captions when generating the next token
                                    logits_input_ids = masked_token_ids[:, :cur_start + step + 1]
                                else:  # less diverse
                                    # input generated tokens of current captions when generating the next token
                                    logits_input_ids = masked_token_ids[:, cur_start:cur_start + step + 1]
                                pred_token_probs, pred_token_ids = custom_generate(config,
                                         pred_scores[:, lang_start + cur_start + step, :], logits_input_ids,
                                        eos_token_id=EOS, pad_token_id=PAD, cluster_index=cluster_index)
                            else:
                                pred_probs = F.softmax(pred_scores[:, lang_start + cur_start + step, :], dim=-1)
                                pred_token_probs, pred_token_ids = pred_probs.max(dim=-1)

                            masked_token_ids[:, cur_start + step] = pred_token_ids
                            if pred_token_ids.sum() == (EOS * len(pred_token_ids)):  # all captions are ending
                                break

                    pred_list.append(masked_token_ids[:, cur_start:cur_start+high].cpu().numpy())  # 5 * (N, L)

                    cur_start += (high + 1)

            else:
                cur_start = 0
                for i in range(config.num_sample_captions):
                    high = config.text_length
                    attention_mask[:, lang_start+cur_start:lang_start+cur_start+high+1] = 1  # attend to current caption

                    pred_scores = generator(
                        grid_features, masked_token_ids, token_type_ids,
                        position_ids, attention_mask,
                        visual_mask=None, cluster_index=cluster_index)

                    pred_scores = pred_scores[0]

                    if config.custom_generation and config.num_beams == 1:
                        # autoregressive: one next token for each caption in a batch
                        # non-autoregressive: all tokens in current generating caption are next tokens
                        # since the sampling strategy codes are implemented for autoregressive models
                        # I reshape all tokens of non-autoregressive models for adaptation
                        current_tokens_scores = pred_scores[:, lang_start+cur_start:lang_start+cur_start+high, :]
                        batch_size, length, vocab_size = current_tokens_scores.shape
                        next_token_scores = current_tokens_scores.reshape(-1, vocab_size)
                        if config.consider_generated_sentence:  # more diverse
                            # input all previous generated captions when generating the next token
                            logits_input_ids = masked_token_ids[:, :cur_start + high]
                        else:  # less diverse
                            # input generated tokens of current captions when generating the next token
                            logits_input_ids = masked_token_ids[:, cur_start:cur_start + high]
                        logits_input_ids = logits_input_ids.reshape(-1)
                        pred_token_probs, pred_token_ids = custom_generate(config, next_token_scores, logits_input_ids,
                                        eos_token_id=EOS, pad_token_id=PAD, cluster_index=cluster_index)
                        pred_token_probs = pred_token_probs.reshape(batch_size, length)
                        pred_token_ids = pred_token_ids.reshape(batch_size, length)
                    else:
                        pred_probs = F.softmax(pred_scores[:, lang_start+cur_start:lang_start+cur_start+high, :],
                                               dim=-1)
                        pred_token_probs, pred_token_ids = pred_probs.max(dim=-1)

                    for step in range(1, config.nonautoregressize_steps):
                        num_mask = max(1, int(high * (1.0 - step / config.nonautoregressize_steps)))

                        mask_id = pred_token_probs.topk(num_mask, -1, False, False)[1]
                        mask_id = (mask_id + batch_id * high).view(-1)

                        pred_token_ids.view(-1)[mask_id] = MASK
                        masked_token_ids[:, cur_start:cur_start + high] = pred_token_ids

                        pred_scores = generator(
                            grid_features, masked_token_ids, token_type_ids,
                            position_ids, attention_mask,
                            visual_mask=None, cluster_index=cluster_index)

                        pred_scores = pred_scores[0]

                        if config.custom_generation:
                            current_tokens_scores = pred_scores[:,
                                                    lang_start + cur_start:lang_start + cur_start + high, :]
                            batch_size, length, vocab_size = current_tokens_scores.shape
                            next_token_scores = current_tokens_scores.reshape(-1, vocab_size)
                            input_ids = masked_token_ids[:, :cur_start + high]
                            input_ids = input_ids.reshape(-1)
                            new_token_probs, new_token_ids = \
                                custom_generate(config, next_token_scores, input_ids,
                                        eos_token_id=EOS, pad_token_id=PAD, cluster_index=cluster_index)
                            new_token_probs = new_token_probs.reshape(batch_size, length)
                            new_token_ids = new_token_ids.reshape(batch_size, length)
                        else:
                            pred_probs = F.softmax(pred_scores[:, lang_start+cur_start:lang_start+cur_start+high, :],
                                                   dim=-1)
                            new_token_probs, new_token_ids = pred_probs.max(dim=-1)

                        pred_token_ids.view(-1)[mask_id] = new_token_ids.view(-1)[mask_id]
                        pred_token_probs.view(-1)[mask_id] = new_token_probs.view(-1)[mask_id]
                        pred_token_probs = (pred_token_probs + new_token_probs) / 2
                        # print(tokenizer.decode(pred_token_ids[0].cpu().numpy()))

                    pred_list.append(pred_token_ids.cpu().numpy())  # 5 * (N, L)

                    cur_start += (high + 1)

        image_ids = list(batch[1].cpu().numpy())
        for level, preds_per_level in enumerate(pred_list, 1):
            for batch_id, image_id in enumerate(image_ids):
                pred_per_level = tokenizer.decode(preds_per_level[batch_id], end_flags=[EOS])
                pred_per_level = re.sub(r'\b(\w+)( \1\b)+', r'\1', pred_per_level)
                if str(image_id) not in img2cap_pred:
                    img2cap_pred[str(image_id)] = [pred_per_level]
                else:
                    img2cap_pred[str(image_id)].append(pred_per_level)

    logger.info('batch_time: {time:.4f} batch_memory: {memory:.2f}'.format(
        time=(time.time() - end) / iteration,
        memory=torch.cuda.max_memory_allocated() / 1024.0 ** 3))

    return img2cap_pred

def save_pred(img2cap_pred, save_dir, model_name, logger):
    img2cap_pred_path = os.path.join(save_dir, 'caption_results_' + model_name + '_5captions')

    if config.custom_generation == 0:
        img2cap_pred_path += '_vanilla_greedy'
    else:
        img2cap_pred_path += '_history' + str(config.consider_generated_sentence) + '_sample' + str(config.do_sample)
        img2cap_pred_path += '_beam' + str(config.num_beams) + '_group' + str(config.num_beam_groups)

    img2cap_pred_path += '_len' + str(config.text_length)

    if config.inference_postfix is not None:
        img2cap_pred_path += config.inference_postfix
    img2cap_pred_path = img2cap_pred_path + '.json'

    logger.info(f"Saving results to {img2cap_pred_path}")
    with open(img2cap_pred_path, 'w') as f:
        json.dump(img2cap_pred, f)

    return img2cap_pred_path


def test(generator, device, model_path, save_dir, model_name, logger, tokenizer=None):
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

    eval_div_acc(pd_caption=img2cap_pred_path, only_eval_cider=config.only_eval_cider,
                 gt_caption=config.id2captions_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config.merge_from_list(args.opts)
    config.freeze()

    model_name = re.search('(.*).pth', config.model_path.split('/')[-1]).group(1)
    save_dir = os.path.join(config.save_dir, config.exp_id)
    mkdir(save_dir)
    logger = setup_logger("inference", save_dir, 0, filename='infer_model_' + model_name + '.log')
    logger.info("Running with config:\n{}".format(config))

    tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)

    device = torch.device(config.device)

    num_types = config.num_sample_captions + 2  # 2 is for visual type and [PAD] type
    bert_config = BertConfig(type_vocab_size=num_types, num_hidden_layers=config.num_hidden_layers,
                             vocab_size_or_config_json_file=num_tokens)
    bert_config.myconfig = config
    generator = Generator(bert_config)
    generator = generator.to(device)

    test(generator, device, config.model_path, save_dir, model_name, logger, tokenizer=tokenizer)
