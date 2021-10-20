'''
Usage example:
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES="0" python evaluate_images.py
eval_fid 1 eval_is 0 eval_clipscore 1 test_val2014 1 test_num -1
evaluate_images_dir /path/to/images/released_model
img_save_dir /path/to/images/generated_images
'''

import argparse
import json
import re
import numpy as np
import pathlib
import torch
import random
from tqdm import tqdm, trange
import os.path, sys
from config import _C as config
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def eval_rprec(img_save_dir, batch_size=64):
    import clip
    from PIL import Image
    from urllib.parse import unquote
    random.seed(42)
    path = pathlib.Path(img_save_dir)
    files = (list(path.glob('*.jpg')) + list(path.glob('*.png')))
    text_list = []
    image_list = []
    for file in sorted(files, key=lambda file: int(re.findall(r'img(\d+)\.', file.name)[0])):
        # image_id = file.name.split(".")[0]
        text = re.sub(r"img\d+\.", "", file.name).replace(".png", "")
        text_list.append(unquote(text))
        image_list.append(os.path.join(img_save_dir, file.name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=True)

    text_with_prompt = ["A photo depicts " + text for text in text_list]

    all_text_features = []
    for i in trange(0, len(text_list), batch_size, desc='Computing text feature...'):
        text = clip.tokenize(text_with_prompt[i: i + batch_size]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        all_text_features.append(text_features.cpu())

    all_image_features = []
    for i in trange(0, len(image_list), batch_size, desc='Computing image feature...'):
        image_tensor_list = list(map(lambda image_file_path: preprocess(Image.open(image_file_path)).to(device),
                                     image_list[i: i + batch_size]))
        image = torch.stack(image_tensor_list)
        with torch.no_grad():
            image_features = model.encode_image(image)
        all_image_features.append(image_features.cpu())

    all_text_features = torch.cat(all_text_features)
    all_image_features = torch.cat(all_image_features)

    R = np.zeros(len(text_list))
    with torch.no_grad():
        for sample_id in trange(len(text_list)):
            # sample 99 sentences from all sentences except current one.
            negative_sent_indices = [i if i < sample_id else i + 1 for i in
                                     random.sample(range(len(text_list) - 1), 99)]
            all_sent_indices = [sample_id] + negative_sent_indices

            all_sent_indices = torch.tensor(all_sent_indices, dtype=torch.long, device=device)

            text_features = all_text_features[all_sent_indices].to(device)
            image_features = all_image_features[sample_id].unsqueeze(0).to(device)

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

            rank = logits_per_text.softmax(dim=0).argmax().item()
            if rank == 0:
                R[sample_id] = 1

    sum = np.zeros(10)
    np.random.shuffle(R)
    for i in range(10):
        sum[i] = np.average(R[i * 3000:(i + 1) * 3000 - 1])
    R_mean = np.average(sum)
    R_std = np.std(sum)

    return R_mean, R_std


def eval_rprec_hard(img_save_dir, rprec_hard_samples_dir):
    import clip
    import pickle
    from PIL import Image
    from urllib.parse import unquote
    random.seed(42)
    path = pathlib.Path(img_save_dir)
    files = (list(path.glob('*.jpg')) + list(path.glob('*.png')))
    sample_list = {}
    for file in sorted(files, key=lambda file: int(re.findall(r'img(\d+)\.', file.name)[0])):
        image_id = file.name.split(".")[0]
        text = re.sub(r"img\d+\.", "", file.name).replace(".png", "")
        sample_list[image_id] = {"text": "A photo depicts " + unquote(text),
                                 "image": os.path.join(img_save_dir, file.name)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=True)

    SPLITS = ['verb', 'color', 'noun', 'number']
    rpre_hard = 0
    for split in SPLITS:
        with open(os.path.join(rprec_hard_samples_dir, f'{split}.pkl'), 'rb') as data:
            hard_samples = pickle.loads(data.read())
        R = np.zeros(len(hard_samples))
        with torch.no_grad():
            for i, image_id in enumerate(tqdm(hard_samples)):
                image_path = sample_list[image_id]["image"]
                captions = ["A photo depicts " + i['caption'] for i in hard_samples[image_id]]
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                text = clip.tokenize(captions).to(device)
                logits_per_image, logits_per_text = model(image, text)
                rank = logits_per_text.softmax(dim=0).argmax().item()
                if rank == 0:
                    R[i] = 1
        sum = np.zeros(10)
        np.random.shuffle(R)
        for i in range(10):
            sum[i] = np.average(R[i * 100:(i + 1) * 100 - 1])
        R_mean = np.average(sum)
        R_std = np.std(sum)
        rpre_hard += R_mean
        print("CLIP-based R-precision Hard on split {} mean:{:.4f} std:{:.4f}".format(split, R_mean, R_std),
              img_save_dir)
    rpre_hard /= 4

    return rpre_hard


def eval_clipscore(img_save_dir):
    from loss import ClipScorer
    from urllib.parse import unquote
    from PIL import Image
    clip_scorer = ClipScorer()
    path = pathlib.Path(img_save_dir)
    files = (list(path.glob('*.jpg')) + list(path.glob('*.png')))
    text_list = []
    image_list = []
    for file in sorted(files):
        # image_id = file.name.split(".")[0]
        text = re.sub(r"img\d+\.", "", file.name).replace(".png", "")
        text_list.append(unquote(text))
        image_list.append(os.path.join(img_save_dir, file.name))
    chunk_size = 64
    clip_score = 0
    with torch.no_grad():
        for i in trange(0, len(text_list), chunk_size):
            current_image_list = [Image.open(image_path) for image_path in image_list[i: i + chunk_size]]
            current_text_list = text_list[i: i + chunk_size]
            clip_score += torch.sum(clip_scorer(current_image_list, current_text_list).to(dtype=torch.float32))
        clip_score /= len(text_list)

    return clip_score.item()


def eval_fid(gt_img_dir, img_save_dir):
    from evaluation.pytorch_fid import fid_score
    fid_paths = [gt_img_dir, img_save_dir]
    print('Evaluating FID: ', fid_paths)
    fid_value = fid_score.calculate_fid_given_paths(paths=fid_paths)
    print('FID: ', fid_value, fid_paths)

    return fid_value


def eval_is(img_save_dir):
    from evaluation.inception_score import model
    model._init_inception()
    mean_score, std_score = model.get_inception_score(model.get_images(img_save_dir))
    return mean_score, std_score


def move_gt_img(gt_img_dir):
    from sample_captions import get_captions
    captions, image_ids, filenames = get_captions(karpathy_test=config.test_gts_dict,  test_val2014=config.test_val2014)
    for filename_idx, filename in enumerate(filenames):
        cmd = 'sudo cp ' + config.gt_img_dir_val2014 + filename \
              + ' ' + gt_img_dir + '/img' + str(filename_idx + 1) + '.' + filename
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate_images")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config.merge_from_list(args.opts)

    if os.path.exists(config.evaluate_images_dir):  # evaluate specific dir
        img_save_dir = config.evaluate_images_dir
    else:  # evaluate the latest existing dir with respect to exp_id
        model_paths = [os.path.join(config.img_save_dir, f) for f in os.listdir(config.img_save_dir)
                       if os.path.isdir(os.path.join(config.img_save_dir, f)) and (config.exp_id in f)]
        latest_model_path = ''
        latest_iteration = 0
        for model_path in model_paths:
            search_res = re.search(config.exp_id + '_0(.*)', model_path)
            if search_res is not None:
                iteration = search_res.group(1)
                if int(iteration) > latest_iteration:
                    latest_iteration = int(iteration)
                    latest_model_path = model_path
        assert os.path.exists(latest_model_path), "latest_model_path not exists:%s"%latest_model_path
        img_save_dir = latest_model_path

    results = {}
    results['img_save_dir'] = img_save_dir
    results['test_num'] = config.test_num
    if config.test_val2014:
        results['split'] = 'val2014_30000'
    else:
        results['split'] = 'karpathy_test'

    result_path = img_save_dir + '_testval' + str(config.test_val2014)

    if config.eval_rprec:
        print('Evaluating CLIP-based R-precision: ', img_save_dir)
        R_mean, R_std = eval_rprec(img_save_dir)
        print("CLIP-based R-precision mean:{:.4f} std:{:.4f}".format(R_mean, R_std), img_save_dir)
        results['CLIPRPrec_mean'] = str(R_mean)
        results['CLIPRPrec_std'] = str(R_std)
        result_path += '_rprec' + str(round(R_mean*100, 2))


    if config.eval_rprec_hard:
        print('Evaluating CLIP-based R-precision Hard: ', img_save_dir)
        rpre_hard = eval_rprec_hard(img_save_dir, config.rprec_hard_samples_dir)
        print("CLIP-based R-precision Hard mean:{:.4f}".format(rpre_hard), img_save_dir)
        results[f'CLIPRPrecHard'] = str(rpre_hard)
        result_path += '_hard' + str(round(rpre_hard*100, 2))

    if config.eval_clipscore:
        print('Evaluating CLIPScore: ', img_save_dir)
        clip_score = eval_clipscore(img_save_dir)
        print('CLIPScore:', round(clip_score, 2), img_save_dir)
        results['CLIPScore'] = str(clip_score)
        result_path += '_Clip' + str(round(clip_score, 2))

    if config.eval_fid:
        gt_img_dir = config.gt_img_dir_3w if config.test_val2014 else config.gt_img_dir_5k
        if not os.path.isdir(gt_img_dir):
            assert os.path.exists(config.gt_img_dir_val2014), \
                "config.gt_img_dir_val2014 " + config.gt_img_dir_val2014 + 'not exists!'
            os.mkdir(gt_img_dir)
            move_gt_img(gt_img_dir)

        fid_value = eval_fid(gt_img_dir, img_save_dir)
        results['FID'] = str(fid_value)
        result_path += '_F' + str(round(fid_value, 2))

    if config.eval_is:
        print('Evaluating IS: ', img_save_dir)
        mean_score, std_score = eval_is(img_save_dir)
        print('Evaluated IS:', mean_score, std_score, img_save_dir)
        results['IS_mean'] = str(mean_score)
        results['IS_std'] = str(std_score)
        result_path += '_I' + str(round(mean_score, 2))

    result_path += '.json'
    print("Saving results to path: ", result_path)
    print("Results:", results)
    with open(result_path, 'w') as fw:
        json.dump(results, fw)
