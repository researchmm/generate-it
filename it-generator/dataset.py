import json
import os.path as osp
import random
from collections import namedtuple
from pandas import Series
import zipfile
import numpy as np
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pickle

Sample = namedtuple("Sample", ["captions", "image_id"])


class ZipReader(object):
    def __init__(self):
        super(ZipReader, self).__init__()
        self.id_context = Series()
    def read(self, zip_file, image_name, pid):
        key_name = zip_file + '_' + str(pid)
        if key_name in self.id_context:
            with self.id_context[key_name].open(image_name) as f:
                tmp = f.read()
            return tmp
        else:
            if sys.version_info[0] == 3:
                file_handle = zipfile.ZipFile(zip_file, 'r', zipfile.ZIP_LZMA)
            else:
                file_handle = zipfile.ZipFile(zip_file, 'r')
            self.id_context[key_name] = file_handle
            return self.id_context[key_name].read(image_name)


class COCOCaptionDataset(Dataset):

    def __init__(self, root_anno, root_feat, split, config=None, tokenizer=None):
        self.split = split
        self.root_anno = root_anno
        self.root_feat = root_feat
        self.config = config
        self.EOS=tokenizer.convert_tokens_to_ids('.')
        self.MASK=tokenizer.mask_token_id
        self.PAD=tokenizer.pad_token_id
        self.tokenizer=tokenizer
        self.SEP=tokenizer.sep_token_id
        self.CLS=tokenizer.cls_token_id

        if self.split == 'test':
            self.load_fn = self._get_item_infer
            self.build_infer_samples()
        else:
            self.load_fn = self._get_item_train
            self.build_train_samples()

        self.zipreader = ZipReader()

        if self.config.use_grid_cluster:
            with open(self.config.grid_cluster_id_path_train, 'rb') as f:
                self.img_id_to_cluster_id = pickle.load(f)  # len: 82783, MSCOCO train2014
            with open(self.config.grid_cluster_id_path_valid, 'rb') as f:
                self.img_id_to_cluster_id.update(pickle.load(f))  # len: 123287

        if self.config.use_grid_feat:
            self.grid_feat_train = h5py.File(self.config.grid_feat_path_train, 'r')
            self.grid_feat_valid = h5py.File(self.config.grid_feat_path_valid, 'r')

    def load_grid_feat(self, image_id):
        if 'train' in image_id:
            grid_feature = self.grid_feat_train[f'{image_id}/features'][:]
        else:
            grid_feature = self.grid_feat_valid[f'{image_id}/features'][:]
        return grid_feature

    def build_infer_samples(self):
        id2captions_test = self.config.id2captions_test
        test_samples = self.config.test_samples
        if not osp.exists(id2captions_test) or not osp.exists(test_samples):
            samples = list()
            id2captions = dict()
            print("id2captions_test", id2captions_test, "does not exist! Constructing data now.")

            with open(self.config.dataset_coco) as f:
                captions = json.load(f)
                captions = captions['images']
            for item in captions:
                if len(samples) == 5000:  # the size of karpathy test split
                    # since I want to test with training set for debugging
                    # I also limit the size to 5000
                    break
                if item['split'] in self.split:
                    image_id = item['filename'].split('.')[0]
                    samples.append(image_id)
                    image_id = str(int(image_id[-12:]))
                    id2captions[image_id] = list()
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        id2captions[image_id].append({'caption': caption})
            print("dumping id2captions_test to", id2captions_test)
            with open(id2captions_test, 'w') as f:
                json.dump(id2captions, f)
            print("dumping test_samples to", test_samples)
            with open(test_samples, 'w') as f:
                json.dump({'ids': samples}, f)
        else:
            print("Loading test_samples from", test_samples)
            with open(test_samples) as f:
                samples = json.load(f)['ids']

        self.samples = samples

    def build_train_samples(self):
        file_train_data = self.config.file_train_data
        if not osp.exists(file_train_data):
            with open(self.config.dataset_coco) as f:
                captions = json.load(f)
                captions = captions['images']
            print("file_train_data", file_train_data, "does not exist! Constructing data now.")
            samples = list()
            for item in captions:
                if item['split'] in self.split:
                    caption_list = []
                    image_id = item['filename'].split('.')[0]
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        caption = self.tokenizer.encode(caption)
                        if len(caption) > 25:  # hard-code
                            caption = caption[:25]  # hard-code
                        if len(caption_list) == 5:
                            break
                        caption_list.append(caption)
                    sample = Sample(captions=caption_list, image_id=image_id)
                    samples.append(sample)
            torch.save(samples, file_train_data)
            print("Saved file_train_data to", file_train_data)
        else:
            print("Loading file_train_data from", file_train_data)
            samples = torch.load(file_train_data)

        self.samples = samples

    def __getitem__(self, index):
        return self.load_fn(index)

    def _get_item_train(self, index):
        sample = self.samples[index]
        input_token_id_list = sample.captions
        random.shuffle(input_token_id_list)
        id = input_token_id_list[0]

        length = len(id)
        if length > self.config.text_length:
            id = id[:self.config.text_length]  # crop if current caption is too long
            length = len(id)
        offset = self.config.text_length - length
        input_token_id = id + [self.PAD] * (offset + 1)

        masked_token_id = input_token_id.copy()

        # from first token to the first [eos] token
        selected_idx = random.randint(1, length)  # the first token is [CLS], do not mask
        masked_token_id[selected_idx] = self.MASK

        input_token_id = torch.tensor(input_token_id, dtype=torch.long)
        masked_token_id = torch.tensor(masked_token_id, dtype=torch.long)

        if self.config.use_grid_cluster:
            cluster_index = torch.from_numpy(self.img_id_to_cluster_id[sample.image_id])
        else:
            cluster_index = None

        if self.config.use_grid_feat:
            grid_feature = self.load_grid_feat(sample.image_id)
            grid_feature = np.reshape(grid_feature, (self.config.grid_size**2, self.config.grid_feat_dim))
            grid_feature = torch.from_numpy(grid_feature)
        else:
            grid_feature = None

        gt_captions = [torch.Tensor(_).long() for _ in sample.captions]
        gt_captions = pad_sequence(gt_captions, batch_first=True, padding_value=0)

        return cluster_index, input_token_id, masked_token_id, grid_feature, gt_captions

    def _get_item_infer(self, index):
        sample = self.samples[index]
        image_id = torch.tensor(int(sample[-12:]), dtype=torch.long)

        if self.config.use_grid_cluster:
            cluster_index = torch.from_numpy(self.img_id_to_cluster_id[sample])
        else:
            cluster_index = None

        if self.config.use_grid_feat:
            grid_feature = self.load_grid_feat(sample)
            grid_feature = np.reshape(grid_feature, (self.config.grid_size**2, self.config.grid_feat_dim))
            grid_feature = torch.from_numpy(grid_feature)
        else:
            grid_feature = None

        return grid_feature, image_id, cluster_index

    def __len__(self):
        return len(self.samples)


def collate_fn_train(batch):
    batch = list(zip(*batch))
    cluster_index = None if batch[0][0] is None else torch.stack(batch[0], dim=0)
    input_token_id = pad_sequence(batch[1], batch_first=True, padding_value=0)
    masked_token_id = pad_sequence(batch[2], batch_first=True, padding_value=0)
    grid_feature = None if batch[3][0] is None else torch.stack(batch[3], dim=0)
    gt_captions = batch[4]

    return cluster_index, input_token_id, masked_token_id, grid_feature, gt_captions


def collate_fn_test(batch):
    batch = list(zip(*batch))
    grid_feature = None if batch[0][0] is None else torch.stack(batch[0], dim=0)
    image_id = torch.stack(batch[1], dim=0)
    cluster_index = None if batch[2][0] is None else torch.stack(batch[2], dim=0)

    return grid_feature, image_id, cluster_index
