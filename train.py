import argparse
import numpy as np
import yaml
import torch
import os
from utils import wandb_setup, dict2obj
from trainer import TrainerKInterpolator

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=None, help='config file (.yml) containing the hyper-parameters for inference.')
parser.add_argument('--debug', action='store_true', help='if true, model will not be logged and saved')
parser.add_argument('--seed', type=int, help='seed of torch and numpy', default=1)
parser.add_argument('--val_frequency', type=int, help='training data and weights will be saved in this frequency of epoch')

if __name__ == '__main__':

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.config) as f:
        print(f'Using {args.config} as config file')
        config = yaml.load(f, Loader=yaml.FullLoader)

    for arg in vars(args):
        if getattr(args, arg):
            if arg in config['general']:
                print(f'Overriding {arg} from argparse')
            config['general'][arg] = getattr(args, arg)

    if not config['general']['debug']: wandb_setup(config)
    config = dict2obj(config)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{config.general.gpus}'

    config.general.debug = args.debug
    trainer = TrainerKInterpolator(config)
    trainer.run()
