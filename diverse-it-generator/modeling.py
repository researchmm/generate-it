import torch
from torch import nn
from transformers import modeling_bert
import numpy as np

class VLBertEmbeddings(modeling_bert.BertEmbeddings):
    def __init__(self, config):
        super(VLBertEmbeddings, self).__init__(config)

        self.myconfig = config.myconfig

        self.grid_embed = nn.Sequential(
            nn.Linear(self.myconfig.grid_feat_dim, self.myconfig.grid_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.myconfig.grid_feat_dim, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob))

        self.grid_position_embed = nn.Sequential(
            nn.Linear(4, config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.hidden_dropout_prob))

        self.mask_feat = nn.Parameter(torch.zeros(self.myconfig.grid_feat_dim))

        self.visual_pos = self.box_position(grid_size=self.myconfig.grid_size)
        self.visual_pos = torch.from_numpy(self.visual_pos).unsqueeze(0)

    def box_position(self, grid_size=8):
        n_grids = grid_size ** 2
        boxes = np.zeros(shape=(n_grids, 4), dtype=np.float32)
        for i in range(grid_size):
            for j in range(grid_size):
                # pre-normalize (0 ~ 1)
                x0, x1 = j / grid_size, (j + 1) / grid_size
                y0, y1 = i / grid_size, (i + 1) / grid_size
                coordinate = (x0, y0, x1, y1)
                boxes[i * grid_size + j] = coordinate
        return boxes

    def load_state_dict(self, state_dict_path, loc='cpu'):
        state_dict = torch.load(state_dict_path, map_location=loc)
        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        return new_state_dict

    def forward(self, visual_feats, input_token_ids, token_type_ids, position_ids, visual_mask=None):
        if visual_mask is not None:
            visual_feats = torch.where(visual_mask.view(visual_mask.size(0), visual_mask.size(1), 1).bool(),
                                       self.mask_feat.view(1, 1, -1), visual_feats)
        x = self.grid_embed(visual_feats)
        words_embeddings = self.word_embeddings(input_token_ids)  # which word
        words_embeddings = torch.cat((x, words_embeddings), dim=1)

        visual_pos = self.visual_pos.repeat(len(input_token_ids), 1, 1).to(input_token_ids.device)
        y = self.grid_position_embed(visual_pos)
        position_embeddings = self.position_embeddings(position_ids)  # which token position
        position_embeddings = torch.cat((y, position_embeddings), dim=1)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        return self.dropout(self.LayerNorm(embeddings))


class Generator(modeling_bert.BertPreTrainedModel):
    def __init__(self, config):
        super(Generator, self).__init__(config)

        self.encoder = modeling_bert.BertEncoder(config)
        self.embedding_layer = VLBertEmbeddings(config)
        self.head_mask = [None] * config.num_hidden_layers
        # use the same name (classifier) as in pre-trained models
        self.classifier = modeling_bert.BertLMPredictionHead(config)
        config.vocab_size = config.myconfig.grid_cluster_num
        self.img_classifier = modeling_bert.BertLMPredictionHead(config)
        self.myconfig = config.myconfig

        self.apply(self._init_weights)

        # set cluster_codebook after init_weights, otherwise the params of cluster_codebook will be intialized
        centroids = np.load(self.myconfig.grid_cluster_centroids_path)
        centroids = torch.from_numpy(centroids)
        self.cluster_codebook = nn.Embedding.from_pretrained(centroids, freeze=1)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        self.load_state_dict(state_dict, strict=False)
        del state_dict

    def forward(self, grid_features, masked_token_ids, token_type_ids, position_ids,
                attention_mask, visual_mask=None, cluster_index=None):
        if self.myconfig.use_grid_cluster and cluster_index is not None:
            visual_feats = self.cluster_codebook(cluster_index)
        else:
            visual_feats = grid_features

        embeddings = self.embedding_layer(visual_feats, masked_token_ids, token_type_ids,
                                          position_ids, visual_mask=visual_mask)

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = self.encoder(embeddings, attention_mask, self.head_mask)[0]
        return [self.classifier(hidden_states), self.img_classifier(hidden_states)]


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, weight=None, smoothing=0.2):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.weight = weight

    def forward(self, pred, target):
        """
        Args:
            pred: (N, C), float
            target: (N,), long, values in [0, C-1]
        """
        if self.weight is None:
            self.weight = torch.ones(self.classes, dtype=torch.float32,
                                     device=target.device)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        weight = self.weight[target]
        weighted_loss = torch.sum(-true_dist * pred, dim=-1) * weight

        return torch.mean(weighted_loss) * weight.numel() / weight.sum()


class UnlikelihoodLabelSmoothingLoss(nn.Module):
    '''
    refer to https://github.com/facebookresearch/unlikelihood_training/blob/master/custom/candidate_penalty_ce_loss.py
    '''

    def __init__(self, classes, weight=None, config=None, tokenizer=None, smoothing=0.2):
        super(UnlikelihoodLabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - self.smoothing
        self.classes = classes
        self.weight = weight
        self.autoregressive = config.autoregressive
        self.rank_alpha = config.rank_alpha
        self.use_freq_weight = config.use_freq_weight
        self.config = config
        self.EOS=tokenizer.convert_tokens_to_ids('.')
        self.MASK=tokenizer.mask_token_id
        self.PAD=tokenizer.pad_token_id
        self.tokenizer=tokenizer
        self.SEP=tokenizer.sep_token_id

    def forward(self, pred, target, input_token_ids, attention_mask, masked_token_ids):
        """
        Args:
            pred: (N, C), float
            target: (N,), long, values in [0, C-1]
        """

        if self.weight is None:
            self.weight = torch.ones(self.classes, dtype=torch.float32, device=target.device)
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        weight = self.weight[target]
        weighted_loss = torch.sum(-true_dist * pred, dim=-1) * weight

        mle_loss = torch.mean(weighted_loss) * weight.numel() / weight.sum()

        with torch.no_grad():
            # E.g. DABCC | D | EFFGD => {A,B,C} are negative targets.
            mask_position = (masked_token_ids == self.MASK).to(torch.long)

            neg_weights = []
            neg_cands = []
            att_ids = attention_mask * input_token_ids
            # obtain the negative candidate (generated tokens) id (exclude the target id for each sample in a batch)
            for idx in mask_position.nonzero():
                neg_weight = attention_mask[idx[0]] - (att_ids[idx[0]] == att_ids[idx[0], idx[1]].item()).float()
                neg_cand = att_ids[idx[0]] * neg_weight
                if 2 in self.config.penalize_ngram_repetition and idx[1] > 0:
                    # for a sequence "a b e b c d e b [c]" where "[c]" is the current token to be generated
                    # "b" is previous token for 2-gram "b [c]", which also occurs in previous context
                    # with 2-grams "b e" and "b c". Since "c" is the ground truth word in likelihood loss
                    # we only penalize the word "e" for 2-gram repeating unlikelihood loss
                    # otherwise "a b e b c d e b [e]" would have a repeating 2-gram "b e"
                    prev_mask = (neg_cand == neg_cand[idx[1] - 1]).float()
                    next_mask = torch.roll(prev_mask, 1, 0)
                    if 1 in self.config.penalize_ngram_repetition:
                        neg_weight += next_mask * neg_weight
                    else:
                        neg_weight = next_mask * neg_weight
                    neg_cand = att_ids[idx[0]] * neg_weight.to(bool)
                neg_weights.append(neg_weight)
                neg_cands.append(neg_cand)
            neg_weights = torch.stack(neg_weights, dim=0)
            neg_cands = torch.stack(neg_cands, dim=0).long()
            # assign the index of neg_cands in tensor of pred with weights of neg_weight
            if self.use_freq_weight == 0:  # will add the repeated tokens
                negative_targets = torch.zeros_like(pred).scatter_(1, neg_cands, neg_weights)
            else:
                negative_targets = torch.zeros_like(pred).scatter_add_(1, neg_cands, neg_weights)
                if self.use_freq_weight == -1:
                    negative_targets[negative_targets == 0] = 1e6
                    negative_targets = 1 / negative_targets
        # Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))
        one_minus_probs = torch.clamp((1.0 - pred.exp()), min=1e-5)
        custom_loss = -torch.log(one_minus_probs) * negative_targets
        weighted_custom_loss = torch.sum(custom_loss, dim=-1)
        unlikelihood_loss = torch.sum(weighted_custom_loss) / custom_loss.shape[0]

        loss = mle_loss + self.rank_alpha * unlikelihood_loss

        return loss, mle_loss, unlikelihood_loss
