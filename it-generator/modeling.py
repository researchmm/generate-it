import torch
from torch import nn
from transformers.models.lxmert import modeling_lxmert
import numpy as np
from utils.checkpointer import rename_xlxmert_keys


class XLxmertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = modeling_lxmert.LxmertPredictionHeadTransform(config)
        self.linear_feat = nn.Linear(config.hidden_size, config.visual_feat_dim)
        self.out_cluster = nn.Linear(config.visual_feat_dim, config.myconfig.grid_cluster_num)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        feat = self.linear_feat(hidden_states)
        output = self.out_cluster(feat)
        return output


class Generator(modeling_lxmert.LxmertPreTrainedModel):
    def __init__(self, config):
        super(Generator, self).__init__(config)

        self.embedding_layer = modeling_lxmert.LxmertEmbeddings(config)
        self.encoder = modeling_lxmert.LxmertEncoder(config)
        self.pooler = modeling_lxmert.LxmertPooler(config)
        self.classifier = modeling_lxmert.LxmertPreTrainingHeads(
            config, self.embedding_layer.word_embeddings.weight)

        config.vocab_size = config.myconfig.grid_cluster_num
        self.img_classifier = XLxmertVisualObjHead(config)
        self.config = config
        self.myconfig = config.myconfig

        self.visual_pos = self.box_position(grid_size=self.myconfig.grid_size)
        self.visual_pos = torch.from_numpy(self.visual_pos).unsqueeze(0)
        self.mask_feat = nn.Parameter(torch.zeros(self.myconfig.grid_feat_dim))

        self.apply(self._init_weights)

        centroids = np.load(self.myconfig.grid_cluster_centroids_path)
        centroids = torch.from_numpy(centroids)
        self.cluster_codebook = nn.Embedding.from_pretrained(centroids, freeze=1)

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

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        state_dict = rename_xlxmert_keys(state_dict)
        load_results = self.load_state_dict(state_dict, strict=False)
        print(load_results)
        del state_dict

    def forward(self, grid_features, masked_token_ids, attention_mask,
                visual_mask=None, cluster_index=None, words_embeddings=None):

        token_type_ids = torch.zeros_like(masked_token_ids)
        embedding_output = self.embedding_layer(
            input_ids=masked_token_ids, token_type_ids=token_type_ids, inputs_embeds=words_embeddings)
        if self.myconfig.use_grid_cluster and cluster_index is not None:
            visual_feats = self.cluster_codebook(cluster_index)
        else:
            visual_feats = grid_features
        if visual_mask is not None:
            visual_feats = torch.where(visual_mask.view(visual_mask.size(0), visual_mask.size(1), 1).bool(),
                                       self.mask_feat.view(1, 1, -1), visual_feats)

        visual_pos = self.visual_pos.repeat(len(masked_token_ids), 1, 1).to(masked_token_ids.device)

        # always attend to all visual tokens for i2t & t2i
        # language attention mask:
        if self.myconfig.xlxmert_zeroshot:
            # always attend to all masked_token_ids > 0
            lang_attention_mask = masked_token_ids > 0
            extended_attention_mask = lang_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            # for i2t, only attend to the masked word tokens and its preceding word tokens (auto-regressive)
            # for t2i, attend to masked_token_ids > 0
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            extended_attention_mask = attention_mask[:,:,:,visual_feats.shape[1]:]

        hidden_states = self.encoder(
            lang_feats=embedding_output,
            lang_attention_mask=extended_attention_mask,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            visual_attention_mask=None,
            output_attentions=None,
        )

        visual_output = hidden_states[0][0][-1]
        lang_output = hidden_states[1][0][-1]
        pooled_output = self.pooler(lang_output)

        hidden_states = torch.cat((visual_output, lang_output), dim=1)
        # lang_prediction_scores, cross_relationship_score = self.classifier(lang_output, pooled_output)
        visual_lang_prediction_scores, cross_relationship_score = self.classifier(hidden_states, pooled_output)
        return [visual_lang_prediction_scores, self.img_classifier(hidden_states).float()]
