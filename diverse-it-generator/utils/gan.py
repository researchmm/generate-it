import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class SPADE(nn.Module):
    def __init__(self, x_dim, y_mod_dim=128, norm_type='instance', ks=3):
        """Modified from https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py"""
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(x_dim, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(x_dim, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(y_mod_dim, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(nhidden, x_dim, kernel_size=ks, padding=pw)
        self.beta = nn.Conv2d(nhidden, x_dim, kernel_size=ks, padding=pw)

    def forward(self, x, y):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # y = F.interpolate(y, size=x.size()[2:], mode='nearest')
        y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)
        actv = self.shared(y)
        gamma = self.gamma(actv)
        beta = self.beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=False):
        if noise:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
            return image + self.weight * noise
        else:
            return image


class GeneratorResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, y_mod_dim=128, upscale=True, norm_type='spade_in',
                 SN=nn.utils.spectral_norm):
        super().__init__()
        if upscale:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            upsample = nn.Identity()

        norm = {
            'spade_in': partial(SPADE, norm_type='instance'),
        }[norm_type]

        self.cbn1 = norm(n_in, y_mod_dim=y_mod_dim)
        self.relu1 = nn.LeakyReLU(0.2)
        self.upsample = upsample
        self.conv1 = SN(nn.Conv2d(n_in, n_out, 3, padding=1))
        self.noise1 = NoiseInjection()
        self.cbn2 = norm(n_out, y_mod_dim=y_mod_dim)
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv2 = SN(nn.Conv2d(n_out, n_out, 3, padding=1))
        self.noise2 = NoiseInjection()

        self.res_branch = nn.Sequential(
            upsample,
            SN(nn.Conv2d(n_in, n_out, 1, padding=0))
        )

    def forward(self, x, y=None, noise=False):
        # y: z + coordinate

        h = self.cbn1(x, y)
        h = self.noise1(h, noise)

        h = self.relu1(h)
        h = self.upsample(h)
        h = self.conv1(h)

        h = self.cbn2(h, y)
        h = self.noise2(h, noise)

        h = self.relu2(h)
        h = self.conv2(h)

        res = self.res_branch(x)

        out = h + res

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, target_size=224):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, 3, 3, padding=1)
        # self.tanh = nn.Tanh()
        self.up = nn.Upsample(
            size=(target_size, target_size),
            mode='bilinear', align_corners=False)

    def forward(self, x, up=True):
        h = self.conv(x)
        # h = self.tanh(h)
        if up:
            out = self.up(h)
            return out
        return h


class Generator(nn.Module):
    def __init__(self, emb_dim=2048, base_dim=32, target_size=256, extra_layers=0,
                 init_H=8, init_W=8, norm_type='spade_in', SN=True, codebook_dim=256):
        super().__init__()

        self.init_H = init_H
        self.init_W = init_W
        self.target_size = target_size
        self.norm_type = norm_type
        self.SN = SN
        self.emb_dim = emb_dim

        self.bottleneck_emb = nn.Sequential(
            nn.Conv2d(emb_dim, codebook_dim, 1, padding=0),
            nn.Tanh()
        )

        if SN:
            SN = nn.utils.spectral_norm
        else:
            def SN(x): return x

        upscale_ratio = target_size // init_H

        # n_init = base_dim * upscale_ratio

        resolution_channels = {
            7: min(512, base_dim),
            14: min(512, base_dim),
            28: min(512, base_dim),
            56: min(512, base_dim),
            112: min(256, base_dim),
            224: min(128, base_dim),

            8: min(512, base_dim),
            16: min(512, base_dim),
            32: min(512, base_dim),
            64: min(512, base_dim),
            128: min(256, base_dim),
            256: min(128, base_dim),
        }
        n_init = base_dim

        self.learned_init_conv = nn.Sequential(
            SN(nn.Conv2d(codebook_dim, n_init, 3, padding=1, groups=4)),
        )

        mod_dim = n_init
        self.style_init_conv = nn.Sequential(
            SN(nn.Conv2d(codebook_dim, mod_dim, 3, padding=1, groups=4)),
        )

        n_upscale_resblocks = int(np.log2(upscale_ratio))
        resblocks = []

        to_RGB_blocks = []
        res = init_H
        # upscale resblocks
        for i in range(n_upscale_resblocks):
            n_in = resolution_channels[res]
            res = res * 2
            n_out = resolution_channels[res]

            resblocks.append(GeneratorResidualBlock(n_in, n_out, mod_dim,
                                                    norm_type=norm_type, SN=SN))

            to_RGB_blocks.append(ToRGB(n_out, self.target_size))

        # extra resblocks (no upscales)
        for _ in range(extra_layers):
            n_in = resolution_channels[res]
            # res = res * 2
            n_out = resolution_channels[res]
            resblocks.append(GeneratorResidualBlock(n_in, n_out, mod_dim,
                                                    upscale=False,
                                                    norm_type=norm_type, SN=SN))

            to_RGB_blocks.append(ToRGB(n_out, self.target_size))

        self.resblocks = nn.ModuleList(resblocks)
        self.to_RGB_blocks = nn.ModuleList(to_RGB_blocks)

        self.last = nn.Sequential(
            nn.Tanh(),
        )

        self.init_parameter()

    def forward(self, emb, train=True):
        """
        emb: [B, init_H, init_W, 2048]

        out: [B, 3, target_size, target_size]
        """
        B = emb.size(0)

        if emb.size()[1:] == (self.init_H, self.init_W, self.emb_dim):
            # [B, 2048, init_H, init_W]
            emb = emb.permute(0, 3, 1, 2)

        # [B, n_init, init_H, init_W]
        emb = self.bottleneck_emb(emb)

        h = self.learned_init_conv(emb)
        y = self.style_init_conv(emb)

        out = torch.zeros(B, 3, self.target_size, self.target_size).to(h.device)

        # [B, base_dim, target_size, target_size]
        for i, resblock in enumerate(self.resblocks):

            h = resblock(h, y, noise=train)
            upsample = i+1 < len(self.resblocks)
            RGB_out = self.to_RGB_blocks[i](h, up=upsample)
            out = out + RGB_out

        out = self.last(out)

        return out

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel=32, SN=False):
        super().__init__()

        if SN:
            SN = nn.utils.spectral_norm
        else:
            def SN(x): return x

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            SN(nn.Conv2d(in_channel, channel, 3, padding=1)),
            nn.ReLU(inplace=True),
            SN(nn.Conv2d(channel, in_channel, 1)),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[len("module."):]] = value
    return new_state_dict


class GAN():
    def __init__(self, device, g_ckpt_path):
        G = Generator()  # use default parameters for pretrained model
        # load pretrained Generator model
        g_state_dict = load_state_dict(g_ckpt_path, 'cpu')
        results = G.load_state_dict(g_state_dict, strict=False)
        print('G loaded from', g_ckpt_path)
        print(results)

        self.G = G.to(device)

        for param in self.G.parameters():
            param.requires_grad = False

    def denorm(self, x):
        """(-1, 1) => (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def generate_image(self, visual_feats=None, captions=None, save_dir='img_samples'):
        fake_img = self.G(visual_feats.permute(0, 2, 1).view(len(visual_feats), 2048, 8, 8))
        generated_imgs = self.denorm(fake_img).cpu()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        from torchvision import transforms
        to_pil_image = transforms.ToPILImage()
        if captions is None:
            captions = [i for i in range(len(visual_feats))]
        for caption, img_tensor in zip(captions, generated_imgs):
            img = to_pil_image(img_tensor).convert("RGB")
            fname = f'{caption}.png'
            fpath = os.path.join(save_dir, fname)
            img.save(fpath)
