import logging
import os

import torch
from apex import amp


class Checkpointer(object):

    def __init__(self, model, optimizer=None, scheduler=None,
                 save_dir="", save_to_disk=None, logger=None,amp=None):
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
                        self.model.load_state_dict(checkpoint.pop("model"), strict=True)
                if model_only:
                    checkpoint.pop("optimizer", None)
                    checkpoint.pop("scheduler", None)
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
