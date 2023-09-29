import scipy.io as sio
import numpy as np
import os
import h5py
import torch
from pyexcel_ods import get_data
from termcolor import colored
from scipy.io import savemat
from scipy.ndimage.morphology import distance_transform_edt
from utils.mri_related import fft2c, MulticoilAdjointOp


def multicoil2single(kspace, coilmaps):
    img = MulticoilAdjointOp(center=True, coil_axis=-4, channel_dim_defined=False)(kspace, torch.ones_like(kspace), coilmaps)
    img /= img.abs().max()
    kspace = fft2c(img)
    return kspace, img


class ToTorchIO():
    def __init__(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, sample):
        inputs = []
        outputs = []
        for key in self.input_keys:
            inputs.append(torch.from_numpy(sample[key]))
        for key in self.output_keys:
            outputs.append(torch.from_numpy(sample[key]))
        return inputs, outputs


def load_mat(fn_im_path):
    try:
        f = sio.loadmat(fn_im_path)
    except Exception:
        try:
            f = h5py.File(fn_im_path, 'r')
        except IOError:
            # print("File {} is defective and cannot be read!".format(fn_im_path))
            raise IOError("File {} is defective and cannot be read!".format(fn_im_path))
    return f


def get_valid_slices(valid_slice_info_file, data_type='img', central_slice_only=True):
    valid_slice_info = get_data(valid_slice_info_file)
    ods_col = 2 if data_type == 'img' else 1  # for 'img', ods_col = 2, while for 'dicom', ods_col = 1
    if central_slice_only: ods_col = 3
    valid_slices = {value[0]: list(
        range(*[int(j) - 1 if i == 0 else int(j) for i, j in enumerate(value[ods_col].split(','))])) for value in
        valid_slice_info["Sheet1"][1:] if len(value) != 0}
    return valid_slices


def load_mask(mask_root, nPE, acc_rate, pattern='VISTA'):
    if acc_rate != 1:
        mask_path = os.path.join(mask_root, pattern, f"mask_VISTA_{nPE}x25_acc{acc_rate}_demo.txt")
        mask = np.loadtxt(mask_path, dtype=np.int32, delimiter=",")
    else:
        mask = np.ones((nPE, 25), dtype=np.int32)
    mask = np.transpose(mask)[None, :, None, :]
    return mask


def get_bounding_box_value(path, image_size, offset=10):
    """
    the 8 dims of the box_info are: [xmax, xmin, xmean, xstd, ymax, ymin, ymin, ystd]
    :param path:
    :param image_size:
    :param offset:
    :return:
    """
    box_info = np.load(path)['box_info']
    slc_num = box_info.shape[0]
    xmax = [min(box_info[slc, 0] + offset, image_size[0]) for slc in range(slc_num)]
    xmin = [max(box_info[slc, 1] - offset, 0) for slc in range(slc_num)]
    ymax = [min(box_info[slc, 4] + offset, image_size[1]) for slc in range(slc_num)]
    ymin = [max(box_info[slc, 5] - offset, 0) for slc in range(slc_num)]
    return np.array([xmin, xmax, ymin, ymax])


def generate_weighting_mask(image_size, boundary, mode):
    [xmin, xmax, ymin, ymax] = boundary
    if mode == 'hard':
        box = np.zeros(image_size, dtype=np.uint8)
        box[xmin:xmax, ymin:ymax] = 1
    elif mode == 'exp_decay':
        box = np.ones(image_size, dtype=np.uint8)
        box[xmin:xmax, ymin:ymax] = 0
        box = distance_transform_edt(box) / 4  # laplace weighting decay
        box = np.exp(-0.2*box).astype(np.float32)

    elif mode == 'all':
        box = np.ones(image_size, dtype=np.uint8)
    else:
        raise TypeError('wrong mode type is given')
    return box


def get_bounding_box(box_path, image_size, slc, offset=10, mode='hard'):
    """
    older version: combination of get_bounding_box_value and generate_weighting_mask. And this function can only deal with one single slice.
    :param path:
    :param z:
    :param offset:
    :param mode: mode can be 'hard': within the box = 1, otherwise 0. Or 'exp': within the box = 1, otherwise
                 exponential decay beginning at the boundary
    :return:
    """
    # box_info = np.load(path)['box_info']
    box_info = np.load(box_path)['box_info']
    xmax, xmin = min(box_info[slc, 0] + offset, image_size[0]), max(box_info[slc, 1] - offset, 0)
    ymax, ymin = min(box_info[slc, 4] + offset, image_size[1]), max(box_info[slc, 5] - offset, 0)
    if mode == 'hard':
        box = np.zeros(image_size, dtype=np.uint8)
        box[xmin:xmax, ymin:ymax] = 1
    elif mode == 'exp_decay':
        box = np.ones(image_size, dtype=np.uint8)
        box[xmin:xmax, ymin:ymax] = 0
        box = distance_transform_edt(box) / 4  # laplace weighting decay
        box = np.exp(-0.2*box).astype(np.float32)

    elif mode == 'all':
        box = np.ones(image_size, dtype=np.uint8)
    else:
        raise TypeError('wrong mode type is given')
    return box, [xmin, xmax, ymin, ymax]


def h52mat(load_path, save_path):
    with h5py.File(load_path, 'r') as ds:
        image = np.array(ds['dImgC'])
        image = np.squeeze(image)
        image = np.transpose(image, (3,2,0,1))
        savemat(save_path, {'dImgC': image})
