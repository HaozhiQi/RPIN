import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import RoIAlign

from rpin.utils.config import _C as C
from rpin.models.layers.IN import InterNet
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

        # build image encoder
        self.backbone = build_backbone(C.RPIN.BACKBONE, self.ve_feat_dim, C.INPUT.IMAGE_CHANNEL)

        if C.RPIN.USE_VIN_FEAT or C.RPIN.ROI_MASKING or C.RPIN.ROI_CROPPING:
            self.conv_downsample = nn.Sequential(
                nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )

        # extract object feature -> convert to object state
        pool_size = C.RPIN.ROI_POOL_SIZE
        self.roi_align = RoIAlign(
            (pool_size, pool_size),
            spatial_scale=C.RPIN.ROI_POOL_SPATIAL_SCALE,
            sampling_ratio=C.RPIN.ROI_POOL_SAMPLE_R,
        )

        if C.RPIN.USE_VIN_FEAT:
            roi2state = [nn.Linear(self.ve_feat_dim, self.in_feat_dim * self.num_objs), nn.ReLU()]
        elif C.RPIN.ROI_MASKING or C.RPIN.ROI_CROPPING:
            roi2state = [nn.Linear(self.ve_feat_dim, self.in_feat_dim), nn.ReLU()]
        else:
            roi2state = [nn.Linear(self.ve_feat_dim * pool_size * pool_size, self.in_feat_dim), nn.ReLU()]

        for _ in range(C.RPIN.N_EXTRA_ROI_F):
            roi2state.append(nn.Linear(self.in_feat_dim, self.in_feat_dim))
            roi2state.append(nn.ReLU(inplace=True))
        self.roi2state = nn.Sequential(*roi2state)

        graph = []
        for i in range(self.time_step):
            graph.append(InterNet(self.in_feat_dim))
        self.graph = nn.ModuleList(graph)

        predictor = [nn.Linear(self.in_feat_dim * self.time_step, self.in_feat_dim), nn.ReLU()]
        for _ in range(C.RPIN.N_EXTRA_PRED_F):
            predictor.append(nn.Linear(self.in_feat_dim, self.in_feat_dim))
            predictor.append(nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(*predictor)

        self.decoder_output = 4
        self.bbox_decoder = nn.Linear(self.in_feat_dim, self.decoder_output)

        if C.RPIN.MASK_LOSS_WEIGHT > 0:
            self.mask_decoder = nn.Sequential(
                nn.Linear(self.in_feat_dim, self.in_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.in_feat_dim, self.mask_size * self.mask_size),
                nn.Sigmoid(),
            )

    def forward(self, x, rois, num_rollouts=10, g_idx=None, x_t=None, phase='train'):
        self.num_objs = rois.shape[2]
        # x: (b, t, c, h, w)
        # reshape time to batch dimension
        x = self.extract_object_feature(x, rois)

        bbox_rollout = []
        mask_rollout = []
        state_list = [x[:, i] for i in range(self.time_step)]
        for i in range(num_rollouts):
            c = [self.graph[j](state_list[j], g_idx) for j in range(self.time_step)]
            all_c = torch.cat(c, 2)
            s = self.predictor(all_c)
            bbox = self.bbox_decoder(s)
            bbox_rollout.append(bbox)
            if C.RPIN.MASK_LOSS_WEIGHT:
                mask = self.mask_decoder(s)
                mask_rollout.append(mask)
            state_list = state_list[1:] + [s]

        bbox_rollout = torch.stack(bbox_rollout).permute(1, 0, 2, 3)
        bbox_rollout = bbox_rollout.reshape(-1, num_rollouts, self.num_objs, self.decoder_output)

        if len(mask_rollout) > 0:
            mask_rollout = torch.stack(mask_rollout).permute(1, 0, 2, 3)
            mask_rollout = mask_rollout.reshape(-1, num_rollouts, self.num_objs, self.mask_size, self.mask_size)

        outputs = {
            'boxes': bbox_rollout,
            'masks': mask_rollout,
        }
        return outputs

    def extract_object_feature(self, x, rois):
        # visual feature, comes from RoI Pooling
        # RPIN method:
        if C.RPIN.USE_VIN_FEAT:
            # VIN baseline method
            batch_size, time_step = x.shape[0], x.shape[1]
            x = x.reshape((batch_size * time_step,) + x.shape[2:])  # (b x t, c, h, w)
            x = self.backbone(x)
            x = self.conv_downsample(x)
            if x.shape[-1] != 1:
                x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)
            x = self.roi2state[0:2](x)
            x = x.reshape(batch_size, time_step, self.num_objs, -1)
            x = self.roi2state[2:](x)
        elif C.RPIN.ROI_MASKING or C.RPIN.ROI_CROPPING:
            batch_size = x.shape[0] // self.num_objs
            time_step = x.shape[1]
            x = x.reshape((batch_size * self.num_objs * time_step,) + x.shape[2:])
            x = self.backbone(x)
            x = self.conv_downsample(x)
            if x.shape[-1] != 1:
                x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)
            x = x.reshape(batch_size, self.num_objs, time_step, -1)
            x = x.permute((0, 2, 1, 3))
            x = self.roi2state(x)
        else:
            batch_size, time_step = x.shape[0], x.shape[1]
            x = x.reshape((batch_size * time_step,) + x.shape[2:])  # (b x t, c, h, w)
            x = self.backbone(x)
            roi_pool = self.roi_align(x, rois.reshape(-1, 5))  # (b * t * num_objs, feat_dim)
            x = roi_pool.reshape(batch_size, time_step, self.num_objs, -1)
            x = self.roi2state(x)  # (b, t, num_obj, feat_dim)
        return x
