import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.roi_align import RoIAlign

from neuralphys.utils.config import _C as C
from neuralphys.models.layers.IN import InterNet
from neuralphys.models.backbones.build import build_backbone


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_time_step = C.RPIN.INPUT_SIZE
        self.ve_feat_dim = C.RPIN.VE_FEAT_DIM  # visual encoder feature dimension
        self.in_feat_dim = C.RPIN.IN_FEAT_DIM  # interaction net feature dimension
        self.num_objs = C.RPIN.NUM_OBJS
        self.po_feat_dim = (
            self.in_feat_dim if C.RPIN.COOR_FEATURE_EMBEDDING or C.RPIN.COOR_FEATURE_SINUSOID else 2
        ) if C.RPIN.COOR_FEATURE else 0  # position feature dimension

        # build image encoder
        self.backbone = build_backbone(C.RPIN.BACKBONE, self.ve_feat_dim)
        if C.RPIN.IMAGE_UP:
            self.blur_conv = nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, kernel_size=3, padding=1)

        # extract object feature
        pool_size = C.RPIN.ROI_POOL_SIZE
        spatial_scale = 1.0 / 2.0 if C.RPIN.IMAGE_UP else 1.0 / 4.0
        self.roi_align = RoIAlign((pool_size, pool_size), spatial_scale=spatial_scale, sampling_ratio=1)
        self.fc0 = nn.Linear(self.ve_feat_dim * pool_size * pool_size, self.in_feat_dim)

        # coordinate features
        if C.RPIN.COOR_FEATURE:
            if C.RPIN.COOR_FEATURE_EMBEDDING:
                self.fc0_coor = nn.Linear(2, self.in_feat_dim)
                self.fc1_coor = nn.Linear(self.in_feat_dim, self.in_feat_dim)
            self.red_coor = nn.Linear(self.in_feat_dim + self.po_feat_dim, self.in_feat_dim)

        # interaction networks
        self.temporal_input = 4
        self._init_interaction_net()

        self.decoder_output = 4
        self.aggregator = nn.Linear(self.in_feat_dim * self.temporal_input, self.in_feat_dim)
        self.state_decoder = nn.Linear(self.in_feat_dim, self.decoder_output)

    def _init_interaction_net(self):
        # interaction network
        graph = []
        for i in range(self.temporal_input):
            graph.append(InterNet(self.in_feat_dim))
        self.graph = nn.ModuleList(graph)

    def forward(self, x, rois, src_coor_features=None, num_rollouts=8, data_pred=None, phase=None, ignore_idx=None):
        self.num_objs = rois.shape[2]
        # x: (b, t, c, h, w)
        # reshape time to batch dimension
        time_step = x.shape[1]
        assert self.input_time_step == time_step
        # threshold, used for conditional IN computation
        r = ((rois[..., 4] - rois[..., 2]) / 2 + (rois[..., 3] - rois[..., 1]) / 2) / 2
        r = r.mean(1).detach()

        x = self.extract_object_feature(x, rois)

        # coordinate feature, provided as input
        if C.RPIN.COOR_FEATURE:
            coor_features = src_coor_features[:, :time_step]
            x = self.attach_position_embedding(x, coor_features)

        bbox_rollout = []
        state_list = [x[:, 0], x[:, 1], x[:, 2], x[:, 3]]
        coor_list = [src_coor_features[:, 0], src_coor_features[:, 1], src_coor_features[:, 2], src_coor_features[:, 3]]

        for i in range(num_rollouts):
            c1 = self.graph[0](state_list[0], coor_list[0], r, ignore_idx)
            c2 = self.graph[1](state_list[1], coor_list[1], r, ignore_idx)
            c3 = self.graph[2](state_list[2], coor_list[2], r, ignore_idx)
            c4 = self.graph[3](state_list[3], coor_list[3], r, ignore_idx)
            all_c = torch.cat([c1, c2, c3, c4], 2)
            s = self.aggregator(all_c)
            bbox = self.state_decoder(s)
            bbox_rollout.append(bbox)
            state_list = state_list[1:] + [s]
            coor_list = coor_list[1:] + [bbox[..., 2:]]

        bbox_rollout = torch.stack(bbox_rollout).permute(1, 0, 2, 3)
        bbox_rollout = bbox_rollout.reshape(-1, num_rollouts, self.num_objs, self.decoder_output)

        outputs = {
            'bbox': bbox_rollout
        }
        return outputs

    def attach_position_embedding(self, x, coor_features):
        emb_features = coor_features

        if C.RPIN.COOR_FEATURE_EMBEDDING:
            emb_features = F.relu(self.fc0_coor(emb_features))
            emb_features = F.relu(self.fc1_coor(emb_features))

        x = torch.cat([x, emb_features], dim=-1)
        x = F.relu(self.red_coor(x))
        return x

    def extract_object_feature(self, x, rois):
        # visual feature, comes from RoI Pooling
        batch_size, time_step = x.shape[0], x.shape[1]
        x = x.reshape((batch_size * time_step,) + x.shape[2:])  # (b x t, c, h, w)
        x = self.backbone(x)
        if C.RPIN.IMAGE_UP:
            x = F.interpolate(x, scale_factor=2)
            x = F.relu(self.blur_conv(x))  # (batch x time, c, h, w)
        roi_pool = self.roi_align(x, rois.reshape(-1, 5))  # (b * t * num_objs, feat_dim)
        roi_pool = roi_pool.reshape(batch_size, time_step, self.num_objs, -1)
        x = F.relu(self.fc0(roi_pool))  # (b, t, num_obj, feat_dim)
        return x
