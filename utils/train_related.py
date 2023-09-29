import os
import glob
import torch
import wandb
import numpy as np
from datetime import datetime
import math
from utils import fix_dict_in_wandb_config


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class Logger:
    def __init__(self,
                 best_eval_mode='max',
                 vis_log_video_fps=1,
                 eval_error_map_vmax=0.04):
        self.log_metric_dict = {}
        self.log_img_items = []
        self.log_video_items = []
        self.vis_log_video_fps = vis_log_video_fps
        assert best_eval_mode in ['max', 'min']
        self.best_eval_mode = best_eval_mode
        self.best_eval_result = -np.inf if best_eval_mode == 'max' else np.inf
        self.best_update_flag = False
        self.eval_error_map_vmax = eval_error_map_vmax
        # for vis_item in self.log_vis_items:
        #     exec(f'self.{vis_item} =dict()')

    def update_metric_item(self, item, value):
        if item not in self.log_metric_dict:
            self.log_metric_dict[item] = value
        else:
            self.log_metric_dict[item] += value

    def wandb_log(self, epoch):
        # for item in self.log_img_items:
        #     wandb.log({item: list(eval(f'self.{item}.values()'))}, commit=False)
        # for item in self.log_video_items:
        #     wandb.log({item: list(eval(f'self.{item}.values()'))}, commit=False)
        wandb.log({'epoch': epoch}, commit=False)
        wandb.log(self.log_metric_dict, commit=False)
        wandb.log({'best_eval_results': self.best_eval_result})

    def wandb_log_final(self):
        test_table = wandb.Table(data=self.wandb_infer.data_list, columns=self.wandb_infer.save_table_column)
        wandb.log({'test_table': test_table}, commit=False)

    def get_metric_value(self, item):
        return self.log_metric_dict[item]

    def update_img_item(self, vis_item, subj_name, value):
        if vis_item not in self.log_img_items:
            self.log_img_items.append(vis_item)
            exec(f'self.{vis_item} =dict()')
        eval(f'self.{vis_item}')[subj_name] = wandb.Image(value, caption=subj_name)

    def update_video_item(self, vis_item, subj_name, value):
        if vis_item not in self.log_video_items:
            self.log_video_items.append(vis_item)
            exec(f'self.{vis_item} =dict()')
        eval(f'self.{vis_item}')[subj_name] = wandb.Video(value, caption=subj_name, fps=self.vis_log_video_fps)

    def reset_metric_item(self):
        self.log_metric_dict = dict.fromkeys(self.log_metric_dict, 0)
        self.best_update_flag = False

    def update_best_eval_results(self, currrent_eval_result):
        sign = -1 if self.best_eval_mode == 'max' else 1
        if sign * currrent_eval_result < sign * self.best_eval_result:
            self.best_eval_result = currrent_eval_result
            self.best_update_flag = True


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def wandb_setup(args):

    group = args['network']['which']
    run = wandb.init(project='KInterpolator', entity=args['general']['wandb_entity'], group=group, config=args)
    group_id = args['general']['exp_name']
    wandb.config.update({'group_id' : f"{group_id}"})
    time_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    wandb.run.name = group_id + '_' + time_now
    fix_dict_in_wandb_config(wandb)


def restore_training(model, optimizer, scheduler, args):
    if os.path.isdir(args.restore_ckpt):
        args.restore_ckpt = max(glob.glob(f'{args.restore_ckpt}/*.pth'), key=os.path.getmtime)
    ckpt = torch.load(args.restore_ckpt)

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if args.restore_training:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        args.start_epoch = ckpt['epoch']





