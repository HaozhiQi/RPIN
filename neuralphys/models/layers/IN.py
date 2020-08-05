import torch
from torch import nn
import torch.nn.functional as F

from neuralphys.utils.config import _C as C


class InterNet(nn.Module):
    def __init__(self, in_feat_dim):
        super(InterNet, self).__init__()
        self.in_feat_dim = in_feat_dim
        self.self_dynamics = nn.Linear(self.in_feat_dim, self.in_feat_dim)
        self.rel_dynamics = nn.Linear(self.in_feat_dim * 2, self.in_feat_dim)
        self.affector = nn.Linear(self.in_feat_dim, self.in_feat_dim)
        self.output = nn.Linear(self.in_feat_dim + self.in_feat_dim, self.in_feat_dim)

    def forward(self, x, rois=None, thresh=None, ignore_idx=None):
        # input should be (batch x num_obj x in_feat_dim)
        s = x
        if x.dim() == 2:
            s = s[None, ...]
            x = x[None, ...]
            rois = rois[None, ...]
        batch_size, num_objs = x.shape[:2]
        objs = [x[:, i] for i in range(num_objs)]
        x = F.relu(self.self_dynamics(x))
        r = []
        indicators = []
        for i in range(num_objs):
            for j in range(num_objs):
                if i == j:
                    continue

                if C.RPIN.IN_CONDITION:
                    threshold = (C.RPIN.IN_CONDITION_R * (thresh[:, i] + thresh[:, j])) ** 2
                    indicator = ((rois[:, i, 0] - rois[:, j, 0]) * C.RPIN.INPUT_WIDTH) ** 2 + \
                                ((rois[:, i, 1] - rois[:, j, 1]) * C.RPIN.INPUT_HEIGHT) ** 2 <= threshold
                    indicator = indicator[:, None].detach().float()
                    indicators.append(indicator)

                if ignore_idx is not None:
                    ignore_relation = ignore_idx[:, [i]] * ignore_idx[:, [j]]
                    r.append(torch.cat([objs[i] * ignore_relation, objs[j] * ignore_relation], 1))
                else:
                    r.append(torch.cat([objs[i], objs[j]], 1))

        r = torch.stack(r, 1)
        r = F.relu(self.rel_dynamics(r))
        if len(indicators) > 0:
            indicators = torch.stack(indicators, 1)
            r = r * indicators
        r = r.reshape(batch_size, num_objs, num_objs - 1, self.in_feat_dim)
        r = r.sum(dim=2)
        pred = x + r
        a = F.relu(self.affector(pred))
        a = torch.cat([a, s], 2)
        out = self.output(a)
        return out
