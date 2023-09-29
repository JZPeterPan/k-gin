import os.path

import numpy as np
import medutils
import h5py
import pandas as pd


class ToNpDtype(object):
    def __init__(self, key_val_pair):
        self.key_val_pair = key_val_pair

    def __call__(self, sample):
        for key, val in self.key_val_pair:
            sample[key] = sample[key].astype(val)
        return sample


class MriOp(object):
    def __init__(self, mask, smaps):
        self.mask = mask
        self.smaps = smaps

    def forward(self, x):
        return medutils.mri.mriForwardOp(x, self.smaps, self.mask, soft_sense_dim=0)

    def adjoint(self, x):
        return medutils.mri.mriAdjointOp(x, self.smaps, self.mask, coil_axis=1)


class Normalize(object):
    def __init__(self, mode='2D', scale=1, axis=(-1, -2)):
        assert mode in ('2D', '3D')
        self.mode = mode
        self.scale = scale
        self.axis = axis

    def __call__(self, sample):
        if self.mode == '2D':
            min_2d = np.min(np.abs(sample['reference']), axis=self.axis)
            max_2d = np.max(np.abs(sample['reference']), axis=self.axis)
            for key in ['reference', 'kspace']:
                sample[key] = (sample[key] - min_2d[:, None, None])/((max_2d - min_2d)[:, None, None])
        elif self.mode == '3D':
            max_3d = np.max(np.abs(sample['reference']))
            min_3d = np.min(np.abs(sample['reference']))
            for key in ['kspace', 'reference']:
                sample[key] = (sample[key])/(max_3d - min_3d) * self.scale
        return sample


class LoadMask(object):
    def __init__(self, pattern, R, data_root):
        self.pattern = pattern  # if isinstance(pattern, list) else [pattern]
        self.R = R if isinstance(R, list) else [R]
        self.data_root = data_root

    def __call__(self, sample):
        R = np.random.choice(self.R)
        masks = []
        mask_idx = np.random.choice(np.arange(100))
        if R != 1:
            try:
                mask_path = os.path.join(self.data_root, self.pattern, f"mask_VISTA_{sample['nPE']}x25_acc{R}_{mask_idx}.txt")
                mask = np.loadtxt(mask_path, dtype=np.int32, delimiter=",")
            except:
                print(f"WARNING: mask_VISTA_{sample['nPE']}x25_acc{R}_{mask_idx}.txt not found, use demo mask instead")
                # TODO: We only uploaded ONE demo mask, more masks needed to be generated for the training
                mask_path = os.path.join(self.data_root, self.pattern, f"mask_VISTA_{sample['nPE']}x25_acc{R}_demo.txt")
                mask = np.loadtxt(mask_path, dtype=np.int32, delimiter=",")
        else:
            mask = np.ones((sample['nPE'], 25))  #, dtype=np.int32)
        masks.append(np.transpose(mask)[None, :, None, :])
        sample['mask'] = np.concatenate(masks, 0)
        sample['R'] = R
        return sample


class ExtractTimePatch(object):
    def __init__(self, patch_size, keys_dim_pair, mode='val'):
        self.patch_size = patch_size
        self.keys_dim_pair = keys_dim_pair
        self.mode = mode

    def __call__(self, sample):
        frames = sample[self.keys_dim_pair[0][0]].shape[self.keys_dim_pair[0][1]]
        if self.mode == 'train':
            start_idx = np.random.randint(0, frames-self.patch_size if frames-self.patch_size > 0 else 1)
        else:
            start_idx = 0

        for key, dim in self.keys_dim_pair:
            sample[key] = np.swapaxes(sample[key], 0, dim)
            sample[key] = sample[key][start_idx:start_idx+self.patch_size]
            sample[key] = np.swapaxes(sample[key], 0, dim)

        return sample
