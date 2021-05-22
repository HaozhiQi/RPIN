import torch
from torch import nn
from rpin.utils.config import _C as C


class InterNet(nn.Module):
    def __init__(self, in_feat_dim):
        super(InterNet, self).__init__()
        self.in_feat_dim = in_feat_dim
        # self dynamics, input object state, output new object state
        self_dynamics = [
            nn.Linear(self.in_feat_dim, self.in_feat_dim), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RPIN.N_EXTRA_SELFD_F):
            self_dynamics.append(nn.Linear(self.in_feat_dim, self.in_feat_dim))
            self_dynamics.append(nn.ReLU(inplace=True))
        self.self_dynamics = nn.Sequential(*self_dynamics)
        # relation dynamics, input pairwise object states, output new object state
        rel_dynamics = [
            nn.Linear(self.in_feat_dim * 2, self.in_feat_dim), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RPIN.N_EXTRA_RELD_F):
            rel_dynamics.append(nn.Linear(self.in_feat_dim, self.in_feat_dim))
            rel_dynamics.append(nn.ReLU(inplace=True))
        self.rel_dynamics = nn.Sequential(*rel_dynamics)
        # affector
        affector = [
            nn.Linear(self.in_feat_dim, self.in_feat_dim), nn.ReLU(inplace=True)
        ]
        for _ in range(C.RPIN.N_EXTRA_AFFECTOR_F):
            affector.append(nn.Linear(self.in_feat_dim, self.in_feat_dim))
            affector.append(nn.ReLU(inplace=True))
        self.affector = nn.Sequential(*affector)
        # aggregator
        aggregator = [nn.Linear(self.in_feat_dim * 2, self.in_feat_dim), nn.ReLU(inplace=True)]
        for _ in range(C.RPIN.N_EXTRA_AGGREGATOR_F):
            aggregator.append(nn.Linear(self.in_feat_dim, self.in_feat_dim))
            aggregator.append(nn.ReLU(inplace=True))
        self.aggregator = nn.Sequential(*aggregator)

    def forward(self, x, g_idx=None):
        s = x
        batch_size, num_objs, _ = x.shape
        x1 = x.repeat(1, num_objs - 1, 1)
        i1 = g_idx[..., [0]].repeat(1, 1, x1.shape[-1])
        y1 = torch.gather(x1, 1, i1)
        i2 = g_idx[..., [1]].repeat(1, 1, x1.shape[-1])
        y2 = torch.gather(x1, 1, i2)
        r = torch.cat([y1, y2], dim=-1)
        r = r * g_idx[:, :, [2]]
        r = r.reshape(r.shape[0], num_objs, num_objs - 1, r.shape[-1])
        r = self.rel_dynamics(r)
        r = r.sum(dim=2)

        x = self.self_dynamics(x)

        pred = x + r
        a = self.affector(pred)
        a = torch.cat([a, s], 2)
        out = self.aggregator(a)
        return out
