import torch
from torch import nn
import torch.nn.functional as F

from rpin.utils.config import _C as C


class InterNet(nn.Module):
    def __init__(self, in_feat_dim):
        super(InterNet, self).__init__()
        self.in_feat_dim = in_feat_dim
        # self dynamics, input object state, output new object state
        self_dynamics = [
            nn.Conv2d(self.in_feat_dim, self.in_feat_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RPIN.N_EXTRA_SELFD_F):
            self_dynamics.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                           kernel_size=C.RPIN.EXTRA_F_KERNEL, stride=1, padding=C.RPIN.EXTRA_F_PADDING))
            self_dynamics.append(nn.ReLU(inplace=True))
        self.self_dynamics = nn.Sequential(*self_dynamics)
        # relation dynamics, input pairwise object states, output new object state
        rel_dynamics = [
            nn.Conv2d(self.in_feat_dim * 2, self.in_feat_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RPIN.N_EXTRA_RELD_F):
            rel_dynamics.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                          kernel_size=C.RPIN.EXTRA_F_KERNEL, stride=1, padding=C.RPIN.EXTRA_F_PADDING))
            rel_dynamics.append(nn.ReLU(inplace=True))
        self.rel_dynamics = nn.Sequential(*rel_dynamics)
        # affector
        affector = [
            nn.Conv2d(self.in_feat_dim, self.in_feat_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RPIN.N_EXTRA_AFFECTOR_F):
            affector.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                      kernel_size=C.RPIN.EXTRA_F_KERNEL, stride=1, padding=C.RPIN.EXTRA_F_PADDING))
            affector.append(nn.ReLU(inplace=True))
        self.affector = nn.Sequential(*affector)
        # aggregator
        aggregator = [
            nn.Conv2d(self.in_feat_dim * 2, self.in_feat_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RPIN.N_EXTRA_AGGREGATOR_F):
            aggregator.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                        kernel_size=C.RPIN.EXTRA_F_KERNEL, stride=1, padding=C.RPIN.EXTRA_F_PADDING))
            aggregator.append(nn.ReLU(inplace=True))
        self.aggregator = nn.Sequential(*aggregator)

    def forward(self, x, g_idx=None):
        s = x
        # of shape (b, o, dim, 7, 7)
        batch_size, num_objs, dim, psz, psz = x.shape
        x1 = x.repeat(1, num_objs - 1, 1, 1, 1)
        i1 = g_idx[..., [0], None, None].repeat(1, 1, dim, psz, psz)
        y1 = torch.gather(x1, 1, i1)
        i2 = g_idx[..., [1], None, None].repeat(1, 1, dim, psz, psz)
        y2 = torch.gather(x1, 1, i2)
        r = torch.cat([y1, y2], dim=2)
        r = r * g_idx[:, :, [2], None, None]
        r = r.reshape(-1, dim * 2, psz, psz)
        r = self.rel_dynamics(r)
        r = r.reshape(batch_size, num_objs, num_objs - 1, dim, psz, psz)
        r = r.sum(dim=2)

        x = self.self_dynamics(x.reshape(-1, dim, psz, psz)).reshape(batch_size, num_objs, dim, psz, psz)

        pred = x + r
        a = self.affector(pred.reshape(-1, dim, psz, psz)).reshape(batch_size, num_objs, dim, psz, psz)
        a = torch.cat([a, s], 2)
        out = self.aggregator(a.reshape(-1, dim * 2, psz, psz)).reshape(batch_size, num_objs, dim, psz, psz)
        return out
