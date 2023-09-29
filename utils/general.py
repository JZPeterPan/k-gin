import numpy as np
import torch
import json
import medutils
import os
import matplotlib.pyplot as plt
import lpips
import os
import json


def fix_dict_in_wandb_config(wandb):
    """"Adapted from [https://github.com/wandb/client/issues/982]"""
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            keys = k.split('.')
            if len(keys) == 2:
                new_key = k.split('.')[0]
                inner_key = k.split('.')[1]
                if new_key not in config.keys():
                    config[new_key] = {}
                config[new_key].update({inner_key: v})
                del config[k]
            elif len(keys) == 3:
                new_key_1 = k.split('.')[0]
                new_key_2 = k.split('.')[1]
                inner_key = k.split('.')[2]

                if new_key_1 not in config.keys():
                    config[new_key_1] = {}
                if new_key_2 not in config[new_key_1].keys():
                    config[new_key_1][new_key_2] = {}
                config[new_key_1][new_key_2].update({inner_key: v})
                del config[k]
            else: # len(keys) > 3
                raise ValueError('Nested dicts with depth>3 are currently not supported!')

    wandb.config = wandb.Config()
    for k, v in config.items():
        wandb.config[k] = v


def normalize_np(img, vmin=None, vmax=None, max_int=255.0):
    """ normalize (magnitude) image
    :param image: input image (np.array)
    :param vmin: minimum input intensity value
    :param vmax: maximum input intensity value
    :param max_int: maximum output intensity value
    :return: normalized image
    """
    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img.copy())
    if vmin == None:
        vmin = np.min(img)
    if vmax == None:
        vmax = np.max(img)
    img = (img - vmin)*(max_int)/(vmax - vmin)
    img = np.minimum(max_int, np.maximum(0.0, img))
    return img


def imsave_custom(img, filepath, normalize_img=True, vmax=None):
    """ Save (magnitude) image in grayscale
    :param img: input image (np.array)
    :param filepath: path to file where k-space should be save
    :normalize_img: boolean if image should be normalized between [0, 255] before saving
    """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img)

    if normalize_img:
        img = normalize_np(img, vmax)
    plt.imsave(filepath, img)


def get_metric(name):
    if name == 'PSNR':
        return medutils.measures.psnr
    elif name == 'SSIM':
        return medutils.measures.ssim
    elif name == 'NRMSE':
        return medutils.measures.nrmseAbs
    elif name == 'LPIPS':
        return lpips.LPIPS(net='alex')
    else:
        raise NotImplementedError


def pad(inp, divisor=8):
    pad_x = int(np.ceil(inp.shape[-2] / divisor)) * divisor - inp.shape[-2]
    pad_y = int(np.ceil(inp.shape[-1] / divisor)) * divisor - inp.shape[-1]
    inp = torch.nn.functional.pad(inp, (pad_y, 0, pad_x, 0))
    return inp, {'pad_x': pad_x, 'pad_y': pad_y}


def unpad(inp, pad_x, pad_y):
    return inp[..., pad_x:, pad_y:]


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def image_normalization(image, scale=1, mode='2D'):
    if mode == '2D':
        return scale * (image - np.min(image)) / (np.max(image) - np.min(image))
    elif mode == '3D':
        if np.iscomplexobj(image):
            image = np.abs(image)
        max_3d = np.max(image)
        min_3d = np.min(image)
        return scale * (image - min_3d) / (max_3d - min_3d)
    else:
        raise NotImplementedError


def image_normalization_torch(image, scale=1, mode='3D'):
    if mode == '3D':
        max_3d = image.abs().max()
        min_3d = image.abs().min()
        image = (image - min_3d) / (max_3d - min_3d) * scale
    else:
        raise NotImplementedError
    return image


def crop_center2d_torch(imgs, crop_size_x_y):
    cropx, cropy = crop_size_x_y[0], crop_size_x_y[1]
    assert imgs.shape[-2] >= cropx and imgs.shape[-1] >= cropy
    shape_x, shape_y = imgs.shape[-2], imgs.shape[-1]
    startx = shape_x // 2 - (cropx // 2)
    starty = shape_y // 2 - (cropy // 2)

    imgs = imgs[..., startx:startx + cropx, starty:starty + cropy]
    return imgs


def cal_lstsq_error(ref, recon):
    ref_flat = ref.flatten()
    recon_flat = recon.flatten()

    # calculate the least-squares solution
    recon2ch = np.concatenate((recon_flat.real, recon_flat.imag))
    ref2ch = np.concatenate((ref_flat.real, ref_flat.imag))
    slope, resid = np.linalg.lstsq(np.stack([recon2ch, np.ones_like(recon2ch)], axis=1), ref2ch, rcond=None)[0]

    recon = recon * slope + resid

    error = np.abs(np.abs(recon) - np.abs(ref))
    return error
