import torch
import torchvision
from utils import load_mask, ToTorchIO, multicoil2single
from dataset.transforms import *


class CINE2DTBase(object):
    def __init__(self, config, mode, transform):

        self.mode = mode
        self.dtype = config.dtype
        self.transform = transform
        self.current_epoch = 0

        subjs_csv = pd.read_csv(eval(f'config.{mode}_subjs_csv'))

        self.data_names = [fname for fname in subjs_csv.filename]
        self.data_names = [fname.split('.')[0] for fname in self.data_names]
        self.data_paths = [os.path.join(config.data_root, f'{name}.h5') for name in self.data_names]
        self.remarks = [fname for fname in subjs_csv.Remarks]

        self.valid_slices = [np.arange(s_start, s_end) for s_start, s_end in zip(eval('subjs_csv.Valid_slice_start'), eval('subjs_csv.Valid_slice_end'))]
        self.data_nPE = [nPE for nPE in subjs_csv.nPE]
        self.data_nFE = [nFE for nFE in subjs_csv.nFE]
        self.masks = {name: load_mask(config.mask_root, nPE, config.acc_rate[0]) for name, nPE in zip(self.data_names, self.data_nPE)}

        self.data_list = []
        self._build_data_list()

    def __len__(self):
        return len(self.data_list)

    def _apply_transform(self, sample):
        return self.transform(sample)

    def get_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        sample = self._load_data(idx)
        sample['epoch'] = self.current_epoch
        if self.transform:
            sample = self._apply_transform(sample)

        # convert multi-coil to single-coil
        sample[0][0], sample[1][0] = multicoil2single(sample[0][0], sample[0][2])
        sample[0][1] = sample[0][1][0]
        del sample[0][2]

        return sample


class CINE2DT(CINE2DTBase, torch.utils.data.Dataset):
    def __init__(self, config, mode):
        super().__init__(config=config, mode=mode, transform=None)
        self.transform = torchvision.transforms.Compose(self.get_transform(config=config))

    def _build_data_list(self):
        for i, subj_name in enumerate(self.data_names):
            for slc in self.valid_slices[i]:
                d = {'subj_name': subj_name,
                     'data_path': self.data_paths[i],
                     'slice': slc,
                     'nPE': self.data_nPE[i],
                     'remarks': self.remarks[i],
                     }
                self.data_list.append(d)

    def _load_data(self, idx):
        data = self.data_list[idx]
        subj_name = data['subj_name']
        # print(f' Loading Data {subj_name}')
        slice = data['slice']
        with h5py.File(data['data_path'], 'r', swmr=True, libver='latest') as ds:
            d = {
                 'kspace': ds['kSpace'][slice].astype(eval(f'np.{self.dtype}')).transpose(0, 1, 3, 2),
                 'smaps': ds['dMap'][slice].astype(eval(f'np.{self.dtype}')).transpose(0, 1, 3, 2),
                 'reference': ds['dImgC'][slice].astype(eval(f'np.{self.dtype}')).transpose(0, 1, 3, 2).squeeze(),
                 'subj_name': subj_name,
                 'nPE': data['nPE'],
                 'slice': slice,
                 'mask': self.masks[subj_name],
                 'remarks': data['remarks']
                 }
        return d

    def get_transform(self, config):
        assert self.mode in ('train', 'val', 'infer')

        data_transforms = []
        if self.mode == 'train':
            data_transforms.append(LoadMask(config.mask_pattern, config.acc_rate, config.mask_root))
        data_transforms.append(Normalize(mode='3D', scale=1.0))
        data_transforms.append(ToNpDtype([('reference', np.complex64), ('kspace', np.complex64),
                                          ('smaps', np.complex64), ('mask', np.float32), ]))
        data_transforms.append(ExtractTimePatch(config.training_patch_time, [('reference', 0), ('kspace', 1), ('mask', 1)], mode=self.mode))
        input_convert_list = ['kspace', 'mask', 'smaps']
        data_transforms.append(ToTorchIO(input_convert_list, ['reference']))

        return data_transforms



