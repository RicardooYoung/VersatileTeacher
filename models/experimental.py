import math
from re import T

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([
            nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, x):
        x = x.neg() * ctx.const
        return x, None

    def grad_reverse(x, const):
        return GRL.apply(x, const)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        y = self.block(x)
        y += x
        return y


class ImageDomainClassifier(nn.Module):
    def __init__(self, input_channels, size, const):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, 1, kernel_size=1)
        )
        # self.conv = ResBlock(input_channels, input_channels)
        self.flatten = nn.Flatten(start_dim=2)
        self.fc = nn.Linear(size * size, 2)
        # self.fc = nn.Sequential(
        # nn.ReLU(inplace=True),
        # nn.Linear(size * size, 2)
        # )
        self.const = const

    def forward(self, x):
        if self.training:
            x = GRL.grad_reverse(x, self.const)
            x = self.conv(x)
            x = self.flatten(x)
            x = self.fc(x)
            x = torch.squeeze(x, dim=1)
            return x
        else:
            return torch.zeros(1)


class InstanceDomainClassifier(nn.Module):
    def __init__(self, input_channels, const, shortcut=True):
        super().__init__()
        self.const = const
        self.mask = None
        self.shortcut = shortcut
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, 1, kernel_size=1)
        )
        # self.conv = nn.Sequential(
        #     ResBlock(input_channels, input_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(input_channels, 1, kernel_size=3, padding=1)
        # )

    def forward(self, x):
        if self.training == True:
            x = GRL.grad_reverse(x, self.const)
            if self.mask is not None:
                if self.shortcut:
                    y = torch.mul(x.transpose(0, 1), self.mask).transpose(0, 1)
                    x = x + y
                    y = None
                else:
                    x = torch.mul(x.transpose(0, 1), self.mask).transpose(0, 1)
            x = self.conv(x)
            x = torch.squeeze(x, dim=1)
            return x
        else:
            return torch.zeros(1)

    def set_mask(self, mask):
        self.mask = mask


class IntegratedImageDiscriminator(nn.Module):
    def __init__(self, base_channels, const):
        super().__init__()
        self.const = const
        self.level_1_branch_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_1_branch_2_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_1_branch_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 4, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.fc = nn.Linear(20 * 20, 2)

    def forward(self, x):
        if self.training == True:
            x[0] = GRL.grad_reverse(x[0], self.const)
            x[1] = GRL.grad_reverse(x[1], self.const)
            x[2] = GRL.grad_reverse(x[2], self.const)

            x[0] = self.level_1_branch_1_conv(x[0])
            x[1] = self.level_1_branch_2_conv(x[1])
            x[2] = self.level_1_branch_3_conv(x[2])
            x[0] = torch.cat((x[0], x[1]), dim=1)

            x[0] = self.level_2_conv(x[0])
            x[0] = torch.cat((x[0], x[2]), dim=1)

            x[0] = self.level_3_conv(x[0])
            x[0] = self.flatten(x[0])
            x[0] = self.fc(x[0])
            x[0] = torch.squeeze(x[0], dim=1)
            return x[0]
        else:
            return torch.zeros(1)


class IntegratedInstanceDiscriminator(nn.Module):
    def __init__(self, base_channels, const):
        super().__init__()
        self.const = const
        self.mask = None
        self.level_1_branch_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_1_branch_1_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_1_branch_2_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_1_branch_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 4, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_2_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_3_conv_seq = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        if self.training == True:
            x[0] = GRL.grad_reverse(x[0], self.const)
            x[1] = GRL.grad_reverse(x[1], self.const)
            x[2] = GRL.grad_reverse(x[2], self.const)

            x[0] = self.level_1_branch_1_conv(x[0])
            if self.mask is not None:
                y = torch.mul(x[0].transpose(0, 1), self.mask[0]).transpose(0, 1)
                x[0] = x[0] + y
                y = None
            x[0] = self.level_1_branch_1_conv_seq(x[0])
            x[1] = self.level_1_branch_2_conv(x[1])
            x[2] = self.level_1_branch_3_conv(x[2])
            x[0] = torch.cat((x[0], x[1]), dim=1)

            x[0] = self.level_2_conv(x[0])
            if self.mask is not None:
                y = torch.mul(x[0].transpose(0, 1), self.mask[1]).transpose(0, 1)
                x[0] = x[0] + y
                y = None
            x[0] = self.level_2_conv_seq(x[0])
            x[0] = torch.cat((x[0], x[2]), dim=1)

            x[0] = self.level_3_conv(x[0])
            if self.mask is not None:
                y = torch.mul(x[0].transpose(0, 1), self.mask[2]).transpose(0, 1)
                x[0] = x[0] + y
                y = None
            x[0] = self.level_3_conv_seq(x[0])
            return x[0]
        else:
            return torch.zeros(1)

    def set_mask(self, mask):
        self.mask = mask


class CenterAwareModule(nn.Module):
    def __init__(self, base_channels) -> None:
        super().__init__()
        self.ctr_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=int(base_channels / 2), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(base_channels / 2), out_channels=int(base_channels / 4), kernel_size=3,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(base_channels / 4), out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.training == True:
            with torch.no_grad():
                ctr = self.ctr_conv(x)
                ctr = ctr.squeeze()
                cls = self.cls_conv(x)
                cls = torch.max(cls, dim=1)[0]
            return torch.mul(ctr, cls)
        else:
            return None


class IntegratedDiscriminator(nn.Module):
    def __init__(self, base_channels, const):
        super().__init__()
        self.const = const
        self.level_1_branch_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_1_branch_2_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 2, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_1_branch_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels * 4, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.level_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.fc = nn.Linear(20 * 20, 2)

    def forward(self, x):
        if self.training == True:
            x[0] = GRL.grad_reverse(x[0], self.const)
            x[1] = GRL.grad_reverse(x[1], self.const)
            x[2] = GRL.grad_reverse(x[2], self.const)

            x[0] = self.level_1_branch_1_conv(x[0])
            x[1] = self.level_1_branch_2_conv(x[1])
            x[2] = self.level_1_branch_3_conv(x[2])
            x[0] = torch.cat((x[0], x[1]), dim=1)

            x[0] = self.level_2_conv(x[0])
            x[0] = torch.cat((x[0], x[2]), dim=1)

            x[0] = self.level_3_conv(x[0])
            x[0] = self.flatten(x[0])
            x[0] = self.fc(x[0])
            x[0] = torch.squeeze(x[0], dim=1)
            return x[0]
        else:
            return torch.zeros(1)
