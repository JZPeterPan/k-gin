import torch.nn.functional as F
import torch


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def gradient_central(data):
    paddings_x = (1, 1, 0, 0)
    paddings_y = (0, 0, 1, 1)

    pad_x = F.pad(data, paddings_x, mode='replicate')
    pad_y = F.pad(data, paddings_y, mode='replicate')

    grad_x = pad_x[:, :, :, 2:] - 2 * data + pad_x[:, :, :, :-2]
    grad_y = pad_y[:, :, 2:] - 2 * data + pad_y[:, :, :-2]

    return grad_x, grad_y


def temporal_grad_central(data):
    data = data.permute(3, 1, 2, 0)
    padding_t = (1, 1, 0, 0)
    pad_t = F.pad(data, padding_t, mode='replicate')
    grad_t = pad_t[..., 2:] - 2 * data + pad_t[..., :-2]
    grad_t = grad_t.permute(3, 1, 2, 0)
    return grad_t


def total_variation_loss(img, edge_map=None):
    img = img.permute(0, 2, 3, 1)
    if edge_map != None:
        edge_map = 1.0 - edge_map.permute(0, 2, 3, 1)
    else:
        edge_map = torch.ones(img.shape).cuda()
    tv_h1 = (torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2) * edge_map[:, :, 1:, :]).sum()
    tv_w1 = (torch.pow(img[:, 1:, :, :] - img[:, :-1, :, :], 2) * edge_map[:, 1:, :, :]).sum()

    tv_h2 = (torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2) * edge_map[:, :, :-1, :]).sum()
    tv_w2 = (torch.pow(img[:, 1:, :, :] - img[:, :-1, :, :], 2) * edge_map[:, :-1, :, :]).sum()

    return 0.25 * (tv_h1 + tv_w1 + tv_h2 + tv_w2) / edge_map.sum()