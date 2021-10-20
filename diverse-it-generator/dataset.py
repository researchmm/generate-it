import json
import os.path as osp
import random
from collections import namedtuple
import numpy as np
import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pickle

Sample = namedtuple("Sample", ["captions", "image_id"])

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

        if self.split == 'test':
            self.load_fn = self._get_item_infer
            self.build_infer_samples()
        else:
            self.load_fn = self._get_item_train
            self.build_train_samples()

        with open(self.config.grid_cluster_id_path_train, 'rb') as f:
            self.img_id_to_cluster_id = pickle.load(f)
        with open(self.config.grid_cluster_id_path_valid, 'rb') as f:
            self.img_id_to_cluster_id.update(pickle.load(f))

        if self.config.use_grid_feat:
            self.grid_feat_train = h5py.File(self.config.grid_feat_path_train, 'r')
            self.grid_feat_valid = h5py.File(self.config.grid_feat_path_valid, 'r')

    def build_infer_samples(self):
        id2captions_test = self.config.id2captions_test
        test_samples = self.config.test_samples
        if not osp.exists(id2captions_test) or not osp.exists(test_samples):
            with open(self.config.dataset_coco) as f:
                captions = json.load(f)
                captions = captions['images']

            samples = list()
            id2captions = dict()
            for item in captions:
                if len(samples) == 5000:  # the size of karpathy test split
                    break
                if item['split'] in self.split:
                    image_id = item['filename'].split('.')[0]
                    samples.append(image_id)
                    image_id = str(int(image_id[-12:]))
                    id2captions[image_id] = list()
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        id2captions[image_id].append({'caption': caption})

            with open(id2captions_test, 'w') as f:
                json.dump(id2captions, f)
            with open(test_samples, 'w') as f:
                json.dump({'ids': samples}, f)
        else:
            with open(test_samples) as f:
                samples = json.load(f)['ids']

        self.samples = samples

    def build_train_samples(self):
        file_train_data = self.config.file_train_data
        if not osp.exists(file_train_data):
            with open(self.config.dataset_coco) as f:
                captions = json.load(f)
                captions = captions['images']

            samples = list()
            for item in captions:
                if item['split'] in self.split:
                    caption_list = []
                    image_id = item['filename'].split('.')[0]
                    for c in item['sentences']:
                        caption = ' '.join(c['tokens']) + '.'
                        caption = self.tokenizer.encode(caption)
                        if len(caption) > 25:  # hard-coded..
                            caption = caption[:25]
                        if len(caption_list) == 5:
                            break
                        caption_list.append(caption)
                    sample = Sample(captions=caption_list, image_id=image_id)
                    samples.append(sample)

            torch.save(samples, file_train_data)
        else:
            samples = torch.load(file_train_data)

        self.samples = samples

    def __getitem__(self, index):
        return self.load_fn(index)

    def _get_item_train(self, index):
        sample = self.samples[index]
        input_token_id_list = sample.captions

        random.shuffle(input_token_id_list)

        input_token_id_list = input_token_id_list[0:self.config.num_sample_captions]

        input_token_id = []
        segment_type_id  = []
        position_id = []
        ix2high = {}
        ix2length = {}
        for i, id in enumerate(input_token_id_list, 1):  # start from 1, to avoid conflict with the [PAD] value (0)
            length = len(id)
            high = self.config.text_length
            if length > high:
                id = id[:high]  # crop if current caption is too long
                length = len(id)
            ix2high[i] = high
            ix2length[i] = length
            offset = high - length
            input_token_id = input_token_id + id + [self.EOS] * offset + [self.SEP]
            segment_type_id = segment_type_id + [i] * (high + 1)  # do not use [PAD] value (0)
            position_id = position_id + [j for j in range(high + 1)]  # restart from zero for each caption

        mask_idx = random.randint(1, len(input_token_id_list))  # to mask which caption
        mask_start_from = segment_type_id.index(mask_idx)  # where the masked caption start from
        masked_token_id = input_token_id.copy()
        if self.config.autoregressive == 0:
            high = ix2high[mask_idx]
            num_masks = random.randint(max(1, int(0.1 * high)), high)
            selected_idx = random.sample(range(high), num_masks)
            for i in selected_idx:
                masked_token_id[mask_start_from + i] = self.MASK
        else:
            # from the first token to the first [eos] token
            selected_idx = random.randint(0, ix2length[mask_idx] - 1)
            masked_token_id[mask_start_from + selected_idx] = self.MASK

        segment_type_id = torch.tensor(segment_type_id, dtype=torch.long)
        position_id = torch.tensor(position_id, dtype=torch.long)
        input_token_id = torch.tensor(input_token_id, dtype=torch.long)
        masked_token_id = torch.tensor(masked_token_id, dtype=torch.long)
        mask_idx = torch.tensor(mask_idx, dtype=torch.long)

        cluster_index = torch.from_numpy(self.img_id_to_cluster_id[sample.image_id])

        if self.config.use_grid_feat:
            if 'train' in sample.image_id:
                grid_feature = self.grid_feat_train[f'{sample.image_id}/features'][:]
            else:
                grid_feature = self.grid_feat_valid[f'{sample.image_id}/features'][:]
            grid_feature = np.reshape(grid_feature, (self.config.grid_size**2, self.config.grid_feat_dim))
            grid_feature = torch.from_numpy(grid_feature)
        else:
            grid_feature = None

        return cluster_index, input_token_id, masked_token_id, grid_feature, \
               segment_type_id, mask_idx, position_id

    def _get_item_infer(self, index):
        sample = self.samples[index]
        image_id = torch.tensor(int(sample[-12:]), dtype=torch.long)

        if self.config.use_grid_cluster:
            cluster_index = torch.from_numpy(self.img_id_to_cluster_id[sample])
        else:
            cluster_index = None

        if self.config.use_grid_feat:
            if 'train' in sample:
                grid_feature = self.grid_feat_train[f'{sample}/features'][:]
            else:
                grid_feature = self.grid_feat_valid[f'{sample}/features'][:]
            grid_feature = np.reshape(grid_feature, (self.config.grid_size**2, self.config.grid_feat_dim))
            grid_feature = torch.from_numpy(grid_feature)
        else:
            grid_feature = None

        return grid_feature, image_id, cluster_index

    def __len__(self):
        return len(self.samples)


def collate_fn_train(batch):
    batch = list(zip(*batch))
    PAD = 0  # hard-coded
    cluster_index = None if batch[0][0] is None else torch.stack(batch[0], dim=0)

    input_token_id = pad_sequence(batch[1], batch_first=True, padding_value=PAD)
    masked_token_id = pad_sequence(batch[2], batch_first=True, padding_value=PAD)

    grid_feature = None if batch[3][0] is None else torch.stack(batch[3], dim=0)

    segment_type_id = pad_sequence(batch[4], batch_first=True, padding_value=PAD)
    mask_idx = torch.stack(batch[5], dim=0)
    position_id = pad_sequence(batch[6], batch_first=True, padding_value=PAD)

    return cluster_index, input_token_id, masked_token_id, grid_feature,\
           segment_type_id, mask_idx, position_id


def collate_fn_test(batch):
    batch = list(zip(*batch))

    grid_feature = None if batch[0][0] is None else torch.stack(batch[0], dim=0)
    image_id = torch.stack(batch[1], dim=0)
    cluster_index = None if batch[2][0] is None else torch.stack(batch[2], dim=0)

    return grid_feature, image_id, cluster_index
