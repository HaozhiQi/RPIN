# Written by Haozhi Qi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import RoIAlign

from rpin.utils.config import _C as C
from rpin.models.layers.CIN import InterNet
from rpin.models.backbones.build import build_backbone


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # define private variables
        self.time_step = C.RPIN.INPUT_SIZE
        self.ve_feat_dim = C.RPIN.VE_FEAT_DIM  # visual encoder feature dimension
        self.in_feat_dim = C.RPIN.IN_FEAT_DIM  # interaction net feature dimension
        self.num_objs = C.RPIN.MAX_NUM_OBJS
        self.mask_size = C.RPIN.MASK_SIZE
        self.picked_state_list = [0, 3, 6, 9]

        # build image encoder
        self.backbone = build_backbone(C.RPIN.BACKBONE, self.ve_feat_dim, C.INPUT.IMAGE_CHANNEL)

        # extract object feature -> convert to object state
        pool_size = C.RPIN.ROI_POOL_SIZE
        self.roi_align = RoIAlign(
            (pool_size, pool_size),
            spatial_scale=C.RPIN.ROI_POOL_SPATIAL_SCALE,
            sampling_ratio=C.RPIN.ROI_POOL_SAMPLE_R,
        )

        roi2state = [nn.Conv2d(self.ve_feat_dim, self.in_feat_dim, kernel_size=3, padding=1),
                     nn.ReLU(inplace=True)]

        for _ in range(C.RPIN.N_EXTRA_ROI_F):
            roi2state.append(nn.Conv2d(self.ve_feat_dim, self.in_feat_dim,
                                       kernel_size=C.RPIN.EXTRA_F_KERNEL, stride=1, padding=C.RPIN.EXTRA_F_PADDING))
            roi2state.append(nn.ReLU(inplace=True))
        self.roi2state = nn.Sequential(*roi2state)

        graph = []
        for i in range(self.time_step):
            graph.append(InterNet(self.in_feat_dim))
        self.graph = nn.ModuleList(graph)

        predictor = [nn.Conv2d(self.in_feat_dim * self.time_step, self.in_feat_dim, kernel_size=1), nn.ReLU()]

        for _ in range(C.RPIN.N_EXTRA_PRED_F):
            predictor.append(nn.Conv2d(self.in_feat_dim, self.in_feat_dim,
                                       kernel_size=C.RPIN.EXTRA_F_KERNEL, stride=1, padding=C.RPIN.EXTRA_F_PADDING))
            predictor.append(nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(*predictor)

        self.decoder_output = 4
        self.bbox_decoder = nn.Linear(self.in_feat_dim * pool_size * pool_size, self.decoder_output)

        if C.RPIN.MASK_LOSS_WEIGHT > 0:
            self.mask_decoder = nn.Sequential(
                nn.Linear(self.in_feat_dim * pool_size * pool_size, self.in_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_feat_dim, self.mask_size * self.mask_size),
                nn.Sigmoid(),
            )

        if C.RPIN.SEQ_CLS_LOSS_WEIGHT > 0:
            self.seq_feature = nn.Sequential(
                nn.Linear(self.in_feat_dim * pool_size * pool_size, self.in_feat_dim * 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_feat_dim * 4, self.in_feat_dim),
                nn.ReLU(inplace=True),
            )
            self.seq_score = nn.Sequential(
                nn.Linear(self.in_feat_dim * len(self.picked_state_list), 1),
                nn.Sigmoid()
            )

    def forward(self, x, rois, num_rollouts=10, g_idx=None, x_t=None, phase='train'):
        self.num_objs = rois.shape[2]
        # x: (b, t, c, h, w)
        # reshape time to batch dimension
        batch_size, time_step = x.shape[:2]
        assert self.time_step == time_step
        # of shape (b, t, o, dim)
        x = self.extract_object_feature(x, rois)

        bbox_rollout = []
        mask_rollout = []
        state_list = [x[:, i] for i in range(self.time_step)]
        state_list_buffer = [x[:, i] for i in range(self.time_step)]
        for i in range(num_rollouts):
            c = [self.graph[j](state_list[j], g_idx) for j in range(self.time_step)]
            all_c = torch.cat(c, 2)
            s = self.predictor(all_c.reshape((-1,) + (all_c.shape[-3:])))
            s = s.reshape((batch_size, self.num_objs) + s.shape[-3:])
            bbox = self.bbox_decoder(s.reshape(batch_size, self.num_objs, -1))
            if C.RPIN.MASK_LOSS_WEIGHT:
                mask = self.mask_decoder(s.reshape(batch_size, self.num_objs, -1))
                mask_rollout.append(mask)
            bbox_rollout.append(bbox)
            state_list = state_list[1:] + [s]
            state_list_buffer.append(s)

        seq_score = []
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT > 0:
            # (p_l * b, o, feat, psz, psz)
            state_list_buffer = torch.cat([state_list_buffer[pid] for pid in self.picked_state_list])
            # (p_l, b, o, feat)
            seq_feature = self.seq_feature(state_list_buffer.reshape(
                len(self.picked_state_list) * batch_size, self.num_objs, -1)
            ).reshape(len(self.picked_state_list), batch_size, self.num_objs, -1)
            valid_seq = g_idx[:, ::self.num_objs - 1, [2]]
            valid_seq = valid_seq[None]
            # (p_l, b, feat)
            seq_feature = (seq_feature * valid_seq).sum(dim=-2) / valid_seq.sum(dim=-2)
            seq_feature = seq_feature.permute(1, 2, 0).reshape(batch_size, -1)
            seq_score = self.seq_score(seq_feature).squeeze(1)

        bbox_rollout = torch.stack(bbox_rollout).permute(1, 0, 2, 3)
        bbox_rollout = bbox_rollout.reshape(-1, num_rollouts, self.num_objs, self.decoder_output)

        if len(mask_rollout) > 0:
            mask_rollout = torch.stack(mask_rollout).permute(1, 0, 2, 3)
            mask_rollout = mask_rollout.reshape(-1, num_rollouts, self.num_objs, self.mask_size, self.mask_size)

        outputs = {
            'boxes': bbox_rollout,
            'masks': mask_rollout,
            'score': seq_score,
        }
        return outputs

    def extract_object_feature(self, x, rois):
        # visual feature, comes from RoI Pooling
        batch_size, time_step = x.shape[0], x.shape[1]
        x = x.reshape((batch_size * time_step,) + x.shape[2:])  # (b x t, c, h, w)
        x = self.backbone(x)
        roi_pool = self.roi_align(x, rois.reshape(-1, 5))  # (b * t * num_objs, feat_dim)
        roi_pool = self.roi2state(roi_pool)
        x = roi_pool.reshape((batch_size, time_step, self.num_objs) + (roi_pool.shape[-3:]))
        return x
