import argparse
import json
import re
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils.logger import setup_logger
from utils.eval_diversity import get_div_n

def eval_div_acc(pd_caption, only_eval_cider, gt_caption):
    model_name = re.search('caption_results_(.*).json', pd_caption).group(1)
    save_dir = pd_caption[:pd_caption.rfind('/')]
    logger = setup_logger("evaluate", save_dir, 0, filename='eval_' + model_name + '.log')
    ptb_tokenizer = PTBTokenizer()

    if only_eval_cider:
        scorers = [(Cider(), "C")]
    else:
        scorers = [(Cider(), "C"),
                   (Spice(), "S"),
                   (Bleu(4), ["B1", "B2", "B3", "B4"]),
                   (Meteor(), "M"), (Rouge(), "R")
                   ]

    logger.info(f"loading ground-truths from {gt_caption}")
    with open(gt_caption) as f:
        gt_captions = json.load(f)
    gt_captions = ptb_tokenizer.tokenize(gt_captions)

    logger.info(f"loading predictions from {pd_caption}")
    with open(pd_caption) as f:
        pred_dict = json.load(f)
    pd_captions = dict()
    if len(pred_dict) > 4:
        for i in range(len(pred_dict['391895'])):
            pd_captions[i] = {}
        for imgid, captions in pred_dict.items():
            for i, caption in enumerate(captions):
                pd_captions[i][imgid] = [{'caption': caption}]
        for level, v in pd_captions.items():
            pd_captions[level] = ptb_tokenizer.tokenize(v)
    else:
        for level, v in pred_dict.items():
            pd_captions[level] = ptb_tokenizer.tokenize(v)

    logger.info("Start evaluating")
    score_all_level = list()
    all_scores = []
    for level, v in pd_captions.items():
        scores = {}
        for (scorer, method) in scorers:
            score, score_list = scorer.compute_score(gt_captions, v)
            if type(score) == list:
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                scores[method] = score
            if method == "C":
                score_all_level.append(np.asarray(score_list))

        if not only_eval_cider:
            logger.info(
                ' '.join([
                    "C: {C:.4f}",
                    "S: {S:.4f}",
                    "M: {M:.4f}", "R: {R:.4f}",
                    "B1: {B1:.4f}", "B2: {B2:.4f}",
                    "B3: {B3:.4f}", "B4: {B4:.4f}"
                ]).format(
                    C=scores['C'],
                    S=scores['S'],
                    M=scores['M'], R=scores['R'],
                    B1=scores['B1'], B2=scores['B2'],
                    B3=scores['B3'], B4=scores['B4']
                ))
        all_scores.append(scores)

    score_all_level = np.stack(score_all_level, axis=1)
    ciders = score_all_level.mean(axis=0)
    if len(ciders) > 4:  # config.num_sample_captions == 5
        logger.info(
            '  '.join([
                "5 level ensemble CIDEr: {C5:.4f}",
                "4 level ensemble CIDEr: {C4:.4f}",
                "3 level ensemble CIDEr: {C3:.4f}",
                "2 level ensemble CIDEr: {C2:.4f}",
            ]).format(
                C5=score_all_level.max(axis=1).mean(),
                C4=score_all_level[:, :4].max(axis=1).mean(),
                C3=score_all_level[:, :3].max(axis=1).mean(),
                C2=score_all_level[:, :2].max(axis=1).mean(),
            ))

        div_1, div_2 = get_div_n(pd_caption)
        logger.info(' '.join(["{model_name}", "div_1: {div_1:.4f}", "div_2: {div_2:.4f}",
                              "C: {C0:.4f}", "{C1:.4f}", "{C2:.4f}", "{C3:.4f}", "{C4:.4f}"]).format(
            model_name=model_name, div_1=div_1, div_2=div_2,
            C0=ciders[0], C1=ciders[1], C2=ciders[2], C3=ciders[3], C4=ciders[4]))

        save_infolder = '_'.join(["{model}", "div{div_1:.1f}", "{div_2:.1f}",
                                   "C{C0:.1f}", "{C1:.1f}", "{C2:.1f}", "{C3:.1f}", "{C4:.1f}"]).format(
            model=model_name[:13], div_1=div_1*100, div_2=div_2*100,
            C0=ciders[0]*100, C1=ciders[1]*100, C2=ciders[2]*100, C3=ciders[3]*100, C4=ciders[4]*100)

        save_outfolder = '_'.join([f"{save_dir}", "{iter}", "div{div_1:.1f}", "{div_2:.1f}",
                                   "C{C_max:.1f}", "{C_mean:.1f}", "{C_min:.1f}"]).format(
            save_dir=save_dir, iter=model_name[7:13], div_1=div_1*100, div_2=div_2*100,
            C_max=ciders.max()*100, C_mean=ciders.mean()*100, C_min=ciders.min()*100)

    else:  # evaluating only one caption
        logger.info(' '.join(["{model_name}", "C: {C0:.4f}"]).format(model_name=model_name, C0=ciders[0]))

        save_infolder = '_'.join(["{model}", "C{C0:.1f}"]).format(model=model_name[:13], C0=ciders[0] * 100)

        save_outfolder = '_'.join([f"{save_dir}", "{iter}", "C{C:.1f}"]).format(
            save_dir=save_dir, iter=model_name[7:13], C=ciders[0] * 100)

    save_infolder = os.path.join(save_dir, save_infolder + '.json')
    with open(save_infolder, 'w') as fw:
        json.dump(all_scores, fw)
    logger.info(f"save_infolder {save_infolder}")

    save_outfolder = save_outfolder + '.json'
    with open(save_outfolder, 'w') as fw:
        json.dump(all_scores, fw)
    logger.info(f"save_outfolder {save_outfolder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--gt_caption", type=str, default='id2captions_test.json')
    parser.add_argument("--res_file", type=str)
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--only_eval_cider", type=bool, default=True)
    args = parser.parse_args()

    save_dir = os.path.join('../../models', args.exp_id)
    pd_caption = os.path.join(save_dir, args.res_file)

    eval_div_acc(pd_caption, args.only_eval_cider, args.gt_caption)
