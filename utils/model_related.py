import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from torchvision.transforms import ColorJitter
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(-1, -2)
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.fc(y).view(b, c, 1)
        return (x * y.expand_as(x)).transpose(-1, -2)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel=1, k_size=5):
        super(eca_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        x = x.transpose(-1, -2)
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return (x * y.expand_as(x)).transpose(-1, -2)


# GenerateMagnitudeVolume pytorch version
def gen_mag_volume_torch(volume):
    if volume.dtype in (torch.complex64, torch.complex128):
        volume_mag = torch.abs(volume).type(torch.float32)
    else:
        raise ValueError('The input volume should be complex type')
    return volume_mag


# MagVolumeNormalize pytorch version
def mag_volume_normalize_torch(volume, mode='2D', scale=255):
    if mode == '2D':
        min_2d = volume.min(-1)[0].min(-1)[0]
        max_2d = volume.max(-1)[0].max(-1)[0]
        volume = (volume - min_2d[..., None, None])/((max_2d - min_2d)[..., None, None])
    elif mode == '3D':
        max_3d = torch.max(volume)
        min_3d = torch.min(volume)
        volume = (volume - min_3d)/(max_3d - min_3d)
    else:
        raise ValueError('The input mode should be 2D or 3D')
    volume = scale * volume
    return volume


# BrightnessAdjustment pytorch version
def brightness_augmentation_torch(volume):
    [_, f, h, w] = volume.size()
    volume = torch.reshape(volume, (1, 1, f * h, w)).repeat_interleave(3, dim=1)
    volume = ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0.5/3.14)(volume.type(torch.uint8)).type(torch.float32)
    volume = torch.reshape(volume[:, 0, ...], (1, f, h, w))
    return volume

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def switch2frameblock_np(img):
    f_pre = np.concatenate((img[-1:, ...], img[:-1, ...]), axis=1)
    f_post = np.concatenate((img[1:, ...], img[:1, ...]), axis=1)
    img = np.concatenate((img, f_pre, f_post), axis=2)
    return img


def switch2frameblock(img):
    f_pre = torch.cat((img[:, -1:, ...], img[:, :-1, ...]), dim=1)
    f_post = torch.cat((img[:, 1:, ...], img[:, :1, ...]), dim=1)
    img = torch.cat((img, f_pre, f_post), dim=2)
    return img


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            if len(dims) == 4:
                self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]
            elif len(dims) == 3:
                self._pad = [pad_wd // 2, pad_wd - pad_wd // 2]

    def pad(self, *inputs, pad_mode='pytorch'):
        if pad_mode == 'pytorch':
            return [F.pad(x, self._pad, mode='replicate') for x in inputs]
        elif pad_mode == 'numpy':
            return [np.pad(x, self._pad, mode='constant') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow(flow, factor, mode='bilinear'):
    new_size = (int(factor * flow.shape[2]), int(factor * flow.shape[3]))
    return factor * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def create_grid(h, w):
    grid_y, grid_x = torch.meshgrid([torch.linspace(0, 1, steps=h),
                                     torch.linspace(0, 1, steps=w),
                                     ])
    return torch.stack([grid_y, grid_x], dim=-1)


def get_2d_fourier_pos_encoding(h, w, feature_size, scale=1., cls_token=False):
    pos_encoding = fourier_pos_encoder_2d(h, w, feature_size, scale=scale)
    pos_encoding = pos_encoding.flatten(0, 1)
    if cls_token:
        pos_encoding = torch.cat([torch.zeros(1, pos_encoding.shape[1]), pos_encoding], dim=0)
    return pos_encoding.unsqueeze(0)


def fourier_pos_encoder_2d(h, w, feature_size, scale=1.):
    B = torch.randn((2, feature_size)) * scale
    grid = create_grid(h, w)
    x_proj = (2. * np.pi * grid) @ B
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
