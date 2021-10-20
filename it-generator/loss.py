import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from pycocoevalcap.cider.cider import Cider
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

CiderD_scorer = Cider()

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def array_to_str(arr, EOS=0):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == EOS:
            break
    return out.strip()


def get_self_critical_reward(greedy_res, gt_ids, gen_result, EOS, seq_per_img=5, repeat_num=0):
    batch_size = len(gen_result)

    greedy_res = greedy_res.data.cpu().numpy()
    gen_result = gen_result.data.cpu().numpy()
    # gt_ids = gt_ids.data.cpu().numpy()

    # {0: [sample_pred0], 1: [sample_pred1], ...,
    # bsz+0: [greedy_pred0], bsz+1: [greedy_pred1], ...}
    res = OrderedDict()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i], EOS)]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i], EOS)]

    # {0: [gt0_1, ..., gt0_5], 1: [gt1_1, ..., gt1_5], ...,
    # bsz+0: [gt0_1, ..., gt0_5], bsz+1: [gt1_1, ..., gt1_5], ...}
    gts = OrderedDict()
    for i in range(batch_size):
        sample_captions = gt_ids[i].data.cpu().numpy()
        gts[i] = [array_to_str(sample_captions[j], EOS) for j in range(seq_per_img)]
        # gts[i] = [array_to_str(gt_ids[i][j], EOS) for j in range(seq_per_img)]
    for i in range(batch_size):
        gts[batch_size + i] = gts[i]

    _, scores = CiderD_scorer.compute_score(gts, res)

    scores = scores[:batch_size] - scores[batch_size:]
    if repeat_num == 0:
        repeat_num = gen_result.shape[1]
    rewards = np.repeat(scores[:, np.newaxis], repeat_num, 1)

    return rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


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


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class ClipScorer:
    def __init__(self, device="cuda:0", prompt="A photo depicts "):
        try:
            import clip
        except:
            raise ImportError("Please install `clip` to use ClipScorer: pip install git+https://github.com/openai/CLIP.git ")
        self.prompt = prompt
        self.device = device
        self.model, self.preprocess_image = clip.load("ViT-B/32", device=self.device, jit=True)
        self.preprocess_tensor = _transform(self.model.input_resolution.item())
        self.tokenize = clip.tokenize

    def to(self, device):
        self.model.to(device)
        self.device = device

    def denorm(self, x):
        """(-1, 1) => (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def __call__(self, image, text):
        assert isinstance(text, (str, list))
        if isinstance(text, str):
            text = [text]
            image = [image]
        text_tensor_list = list(map(lambda x: self.tokenize(self.prompt + x).squeeze(), text))
        text = torch.stack(text_tensor_list).to(self.device)

        if isinstance(image, torch.Tensor):
            image = self.denorm(image)
            image = self.preprocess_tensor(image)
        else:
            image_tensor_list = list(map(lambda x: self.preprocess_image(x), image))
            image = torch.stack(image_tensor_list).to(self.device)
        # with torch.no_grad():
        logits_per_image, logits_per_text = self.model(image, text)
        clipscore = torch.diagonal(logits_per_image, 0)
        clipscore = 2.5 * torch.where(clipscore > 0, clipscore, torch.zeros_like(clipscore))
        return clipscore


# if __name__=="__main__":
#     from PIL import Image
#     import requests
#     import tempfile
#     image_data = requests.get("https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png")
#     with tempfile.NamedTemporaryFile() as f:
#         f.write(image_data.content)
#         image = Image.open(f.name)
#     clip_scorer = ClipScorer()
#     text = "a dog."
#     image_list = [image] * 5
#     text_list = [text] * 5
#     clip_score = clip_scorer(image=image, text=text)
#     print(clip_score)