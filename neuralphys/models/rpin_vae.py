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
        self.fc1 = nn.Linear(self.in_feat_dim, self.in_feat_dim)
        self.fc2 = nn.Linear(self.in_feat_dim, self.in_feat_dim)
        # coordinate features
        if C.RPIN.COOR_FEATURE:
            if C.RPIN.COOR_FEATURE_EMBEDDING:
                self.fc0_coor = nn.Linear(2, self.in_feat_dim)
                self.fc1_coor = nn.Linear(self.in_feat_dim, self.in_feat_dim)
            self.red_coor = nn.Linear(self.in_feat_dim + self.po_feat_dim, self.in_feat_dim)

        # interaction networks
        self.temporal_input = 1
        self.internet = InterNet(self.in_feat_dim)

        self.decoder_output = 4
        self.aggregator = nn.Linear(self.in_feat_dim * self.temporal_input, self.in_feat_dim)
        self.state_decoder = nn.Linear(self.in_feat_dim, self.decoder_output)

        if C.RPIN.VAE:
            self.vae_enc1 = nn.Conv2d(self.ve_feat_dim * 2, self.ve_feat_dim, 3, 2, 1)
            self.vae_enc2 = nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, 3, 2, 1)
            self.vae_enc3 = nn.Conv2d(self.ve_feat_dim, self.ve_feat_dim, 3, 2, 1)
            self.vae_dim = 8
            self.lstm_layers = 1
            self.vae_lstm = nn.LSTM(self.vae_dim, self.vae_dim, self.lstm_layers)
            self.vae_mu_head = nn.Linear(self.ve_feat_dim, 8)
            self.vae_logvar_head = nn.Linear(self.ve_feat_dim, 8)
            self.red_prior = nn.Linear(self.in_feat_dim + 8, self.in_feat_dim)

    def forward(self, x, rois, src_coor_features=None, num_rollouts=8, data_pred=None, phase=None, ignore_idx=None):
        self.num_objs = rois.shape[2]
        # x: (b, t, c, h, w)
        # reshape time to batch dimension
        batch_size, time_step = x.shape[0], x.shape[1]
        assert time_step == 1, 'this model is specifically designed for input time step 1'
        # threshold, used for conditional IN computation
        r = ((rois[..., 4] - rois[..., 2]) / 2 + (rois[..., 3] - rois[..., 1]) / 2) / 2
        r = r.mean(1).detach()

        # visual feature, comes from RoI Pooling
        x = x.reshape((batch_size * time_step,) + x.shape[2:])  # (b x t, c, h, w)
        x = self.backbone(x)
        x_0 = x
        if C.RPIN.IMAGE_UP:
            x = F.interpolate(x, scale_factor=2)
            x = F.relu(self.blur_conv(x))  # (batch x time, c, h, w)
        roi_pool = self.roi_align(x, rois.reshape(-1, 5))  # (b * t * num_objs, feat_dim)
        roi_pool = roi_pool.reshape(batch_size, self.num_objs, -1)
        x = F.relu(self.fc0(roi_pool))  # (b, t, num_obj, feat_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.reshape(batch_size, self.num_objs, self.in_feat_dim)

        bbox_rollout = []

        c = None
        s = x
        if C.RPIN.COOR_FEATURE:
            c = src_coor_features[:, 0]
            s = self.attach_position_embedding(x, c)

        if C.RPIN.VAE:
            if phase == 'train':
                x_t = self.backbone(data_pred.squeeze(1))
                vae_x = torch.cat([x_0, x_t], dim=1)
                vae_x = F.relu(self.vae_enc3(F.relu(self.vae_enc2(F.relu(self.vae_enc1(vae_x))))))
                vae_x = F.adaptive_avg_pool2d(vae_x, (1, 1)).flatten(1)
            else:
                vae_x = x  # just to provide the shape
            z, kl_loss = self.vae_prior(vae_x, num_rollouts, phase)

        for i in range(num_rollouts):
            if C.RPIN.VAE:
                s = self.attach_prior(s, z[i])
            s = self.internet(s, c if C.RPIN.IN_CONDITION else None, r, ignore_idx)
            s = self.aggregator(s)

            bbox = self.state_decoder(s)
            bbox_rollout.append(bbox)

            # attach coordinate feature
            if C.RPIN.COOR_FEATURE or C.RPIN.IN_CONDITION:
                c = bbox[..., 2:]

        bbox_rollout = torch.stack(bbox_rollout).permute(1, 0, 2, 3)
        bbox_rollout = bbox_rollout.reshape(batch_size, num_rollouts, self.num_objs, self.decoder_output)
        outputs = {
            'bbox': bbox_rollout
        }

        if C.RPIN.VAE and phase == 'train':
            outputs['kl_loss'] = kl_loss.unsqueeze(0)

        return outputs

    def attach_position_embedding(self, x, coor_features):
        emb_features = coor_features
        if C.RPIN.COOR_FEATURE_EMBEDDING:
            emb_features = F.relu(self.fc0_coor(emb_features))
            emb_features = F.relu(self.fc1_coor(emb_features))

        x = torch.cat([x, emb_features], dim=-1)
        x = F.relu(self.red_coor(x))

        return x

    def vae_prior(self, x, num_rollouts, phase):
        kl_loss = 0
        if phase == 'train':
            mu = self.vae_mu_head(x)  # batch_size x 8
            logvar = self.vae_logvar_head(x)  # batch_size x 8
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            reparam = mu + eps * std
        else:
            reparam = torch.randn(x.shape[0], self.vae_dim).to('cuda')

        init_hidden = torch.zeros(reparam.shape).to('cuda')[None, :]
        init_input = torch.zeros(reparam.shape).to('cuda').expand(num_rollouts, x.shape[0], self.vae_dim)
        self.vae_lstm.flatten_parameters()
        z, _ = self.vae_lstm(init_input, (init_hidden, reparam[None, :]))

        return z, kl_loss

    def attach_prior(self, x, prior):
        prior = prior[:, None].repeat((1, self.num_objs, 1))
        x = torch.cat([x, prior], dim=-1)
        x = F.relu(self.red_prior(x))
        x = F.normalize(x, dim=-1)
        return x
