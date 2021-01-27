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

        self.image2prior = nn.Sequential(
            nn.Conv2d(self.ve_feat_dim * 2, self.ve_feat_dim, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.vae_dim = 8
        self.lstm_layers = 1
        self.vae_lstm = nn.LSTM(self.vae_dim, self.vae_dim, self.lstm_layers)
        self.vae_mu_head = nn.Linear(self.ve_feat_dim, 8)
        self.vae_logvar_head = nn.Linear(self.ve_feat_dim, 8)
        self.red_prior = nn.Conv2d(self.in_feat_dim + 8, self.in_feat_dim, kernel_size=1)

    def forward(self, x, rois, num_rollouts=10, g_idx=None, x_t=None, phase='train'):
        self.num_objs = rois.shape[2]
        # x: (b, t, c, h, w)
        # reshape time to batch dimension
        batch_size, time_step = x.shape[:2]
        assert self.time_step == time_step
        x, vae_x = self.extract_object_feature(x, x_t, rois, phase)
        z, kl_loss = self.vae_prior(vae_x, num_rollouts, phase)

        bbox_rollout = []
        mask_rollout = []
        state_list = [x[:, i] for i in range(self.time_step)]
        for i in range(num_rollouts):
            state_list = [self.attach_prior(state_list[j], z[i]) for j in range(self.time_step)]
            c = [self.graph[j](state_list[j], g_idx) for j in range(self.time_step)]
            all_c = torch.cat(c, 2)
            s = self.predictor(all_c.reshape((-1,) + (all_c.shape[-3:])))
            s = s.reshape((batch_size, self.num_objs) + s.shape[-3:])
            bbox = self.bbox_decoder(s.reshape(batch_size, self.num_objs, -1))
            bbox_rollout.append(bbox)
            if C.RPIN.MASK_LOSS_WEIGHT:
                mask = self.mask_decoder(s.reshape(batch_size, self.num_objs, -1))
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
            'kl': kl_loss,
        }
        return outputs

    def extract_object_feature(self, x_0, x_t, rois, phase):
        # visual feature, comes from RoI Pooling
        batch_size, time_step = x_0.shape[0], x_0.shape[1]
        x_0 = x_0.reshape((batch_size * time_step,) + x_0.shape[2:])  # (b x t, c, h, w)
        if phase == 'train':
            x_t = x_t.reshape((batch_size * time_step,) + x_t.shape[2:])
            x = self.backbone(torch.cat([x_0, x_t]))
            x, x_t = x[:batch_size], x[batch_size:]
            vae_x = self.image2prior(torch.cat([x, x_t], dim=1)).flatten(1)
        else:
            x = self.backbone(x_0)
            vae_x = x

        roi_pool = self.roi_align(x, rois.reshape(-1, 5))  # (b * t * num_objs, feat_dim)
        roi_pool = self.roi2state(roi_pool)
        x = roi_pool.reshape((batch_size, time_step, self.num_objs) + (roi_pool.shape[-3:]))
        return x, vae_x

    def vae_prior(self, x, num_rollouts, phase):
        kl_loss = 0
        if phase == 'train':
            mu = self.vae_mu_head(x)  # batch_size x 8
            logvar = self.vae_logvar_head(x)  # batch_size x 8
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            reparam = mu + eps * std
            kl_loss = kl_loss.unsqueeze(0)
        else:
            reparam = torch.randn(x.shape[0], self.vae_dim).to('cuda')

        init_hidden = torch.zeros(reparam.shape).to('cuda')[None, :]
        init_input = torch.zeros(reparam.shape).to('cuda').expand(num_rollouts, x.shape[0], self.vae_dim)
        self.vae_lstm.flatten_parameters()
        z, _ = self.vae_lstm(init_input, (init_hidden, reparam[None, :]))

        return z, kl_loss

    def attach_prior(self, x, prior):
        prior = prior[:, None].repeat((1, self.num_objs, 1))
        prior = prior[..., None, None].repeat((1, 1, 1) + x.shape[-2:])
        x = torch.cat([x, prior], dim=2)
        x = x.reshape((x.shape[0] * self.num_objs,) + x.shape[-3:])
        x = F.relu(self.red_prior(x))
        x = x.reshape((-1, self.num_objs,) + x.shape[-3:])
        x = F.normalize(x, dim=2)
        return x
