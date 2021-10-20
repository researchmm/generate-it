import argparse
import json
import re
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils.logger import setup_logger

def eval_captions(pd_caption, gt_caption):
    model_name = re.search('caption_results_(.*).json', pd_caption).group(1) \
        if 'caption_results' in pd_caption else 'debug'
    save_dir = pd_caption[:pd_caption.rfind('/')]
    logger = setup_logger("evaluate", save_dir, 0, filename='eval_' + model_name + '.log')
    ptb_tokenizer = PTBTokenizer()

    scorers = [(Cider(), "C"), (Spice(), "S"), (Bleu(4), ["B1", "B2", "B3", "B4"]), (Meteor(), "M"), (Rouge(), "R")]

    logger.info(f"loading ground-truths from {gt_caption}")
    with open(gt_caption) as f:
        gt_captions = json.load(f)
    gt_captions = ptb_tokenizer.tokenize(gt_captions)

    logger.info(f"loading predictions from {pd_caption}")
    with open(pd_caption) as f:
        pred_dict = json.load(f)

    logger.info("Start evaluating")
    scores = {}
    for (scorer, method) in scorers:
        score, score_list = scorer.compute_score(gt_captions, ptb_tokenizer.tokenize(pred_dict))
        if type(score) == list:
            for m, s in zip(method, score):
                scores[m] = s
        else:
            scores[method] = score

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
    cider = scores['C']
    save_infolder = '_'.join(["{model}", "C{CIDEr:.1f}"]).format(model=model_name[:13], CIDEr=cider * 100)
    save_infolder = os.path.join(save_dir, save_infolder + '.json')
    with open(save_infolder, 'w') as fw:
        json.dump(scores, fw)
    logger.info(f"save_infolder {save_infolder}")

    save_outfolder = '_'.join([f"{save_dir}", "{iter}", "C{C:.1f}"]).format(
        save_dir=save_dir, iter=model_name[7:13], C=cider * 100)
    save_outfolder = save_outfolder + '.json'
    with open(save_outfolder, 'w') as fw:
        json.dump(scores, fw)
    logger.info(f"save_outfolder {save_outfolder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--gt_caption", type=str, default='id2captions_test.json')
    parser.add_argument("--pd_caption", type=str)
    args = parser.parse_args()

    eval_captions(args.pd_caption, args.gt_caption)
