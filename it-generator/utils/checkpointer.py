import logging
import os

import torch
from apex import amp


def rename_xlxmert_keys(state_dict):
    '''
    rename the keys of X-LXMERT's pre-trained model
    to be compatible with the names of labert's model
    '''
    new_keys = []
    del_keys = []
    for key in state_dict:
        if key[:19] == 'module.bert.encoder':
            del_keys.append(key)
            new_keys.append(key[12:])
        elif key[:len('bert.encoder')] == 'bert.encoder':
            del_keys.append(key)
            new_keys.append('encoder' + key[len('bert.encoder'):])

        if key[:len('module.bert.embeddings')] == 'module.bert.embeddings':
            del_keys.append(key)
            new_keys.append('embedding_layer' + key[len('module.bert.embeddings'):])
        elif key[:len('bert.embeddings')] == 'bert.embeddings':
            del_keys.append(key)
            new_keys.append('embedding_layer' + key[len('bert.embeddings'):])

        if key[:len('module.cls')] == 'module.cls':  # pre-trained x-lxmert -> x-lxmert model
            del_keys.append(key)
            new_keys.append('classifier' + key[len('module.cls'):])
        elif key[:len('cls')] == 'cls':  # pre-trained x-lxmert -> x-lxmert model
            del_keys.append(key)
            new_keys.append('classifier' + key[len('cls'):])
        elif key[:len('classifier')] == 'classifier':  # pre-trained bert from labert's bert.pth -> x-lxmert model
            del_keys.append(key)
            new_keys.append('classifier.predictions' + key[len('classifier'):])
        # if key[:22] == 'module.cls.predictions':
        #     del_keys.append(key)
        #     new_keys.append('classifier' + key[22:])
        if key[:23] == 'module.obj_predict_head':
            del_keys.append(key)
            new_keys.append('img_classifier' + key[23:])
        if key[:len('module.mask_feat')] == 'module.mask_feat':
            del_keys.append(key)
            new_keys.append('mask_feat' + key[len('module.mask_feat'):])

        if key[:len('module.bert.pooler')] == 'module.bert.pooler':
            del_keys.append(key)
            new_keys.append('pooler' + key[len('module.bert.pooler'):])
        elif key[:len('bert.pooler')] == 'bert.pooler':
            del_keys.append(key)
            new_keys.append('pooler' + key[len('bert.pooler'):])

    for i, key in enumerate(del_keys):
        state_dict[new_keys[i]] = state_dict[key]
        del state_dict[key]

    for key in list(state_dict.keys()):
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        elif 'beta' in key:
            new_key = key.replace('beta', 'bias')
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    # del state_dict["embedding_layer.token_type_embeddings.weight"]
    return state_dict


class Checkpointer(object):

    def __init__(self, model, optimizer=None, scheduler=None,
                 save_dir="", save_to_disk=None, logger=None, amp=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        self.amp = amp
        if logger is None:
            logger = logging.getLogger("Checkpointer")
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        if not self.save_to_disk:
            return

        data = {}
        if self.model is not None:
            data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        if self.amp is not None:
            data['amp'] = self.amp
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None, model_only=False):
        if f is not None:
            if os.path.exists(f):
                self.logger.info("Loading checkpoint from {}".format(f))
                checkpoint = torch.load(f, map_location=torch.device("cpu"))

                if "model" in checkpoint and self.model:
                    if 'module.' in list(self.model.named_parameters())[0][0]:
                        ckpt_model_dict = checkpoint.pop("model")
                        new_ckpt_model_dict = {"module." + key: value for key, value in ckpt_model_dict.items()}
                        self.model.load_state_dict(new_ckpt_model_dict)
                    else:
                        self.model.load_state_dict(checkpoint.pop("model"), strict=False)
                if model_only:
                    checkpoint.pop("optimizer", None)
                    checkpoint.pop("scheduler", None)
                    if 'Epoch20_LXRT' in f:
                        checkpoint = rename_xlxmert_keys(checkpoint)
                        self.model.load_state_dict(checkpoint, strict=False)
                    return checkpoint

                if "optimizer" in checkpoint and self.optimizer:
                    self.logger.info("Loading optimizer from {}".format(f))
                    self.optimizer.load_state_dict(checkpoint.pop("optimizer"))

                if "scheduler" in checkpoint and self.scheduler:
                    self.logger.info("Loading scheduler from {}".format(f))
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

                if "amp" in checkpoint and self.amp:
                    amp.load_state_dict(checkpoint['amp'])

                return checkpoint
            else:
                self.logger.info("No checkpoint found in {} (path not exists)".format(f))
        else:
            self.logger.info("No checkpoint provided.")
        return {}
