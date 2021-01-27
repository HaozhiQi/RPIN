# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torchvision


class ResNet18(nn.Module):

    def __init__(self, data_format):
        super().__init__()
        self.data_format = data_format
        net = torchvision.models.resnet18(pretrained=False)
        self.num_colors = 7
        conv1 = nn.Conv2d(self.num_colors * data_format, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.reason = nn.Linear(512, 1)
        self.register_buffer('embed_weights', torch.eye(self.num_colors))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([
            net.layer1, net.layer2, net.layer3, net.layer4
        ])

    @property
    def device(self):
        if hasattr(self, 'parameters') and next(self.parameters()).is_cuda:
            return 'cuda'
        else:
            return 'cpu'

    def forward(self, observations):
        assert observations.ndim == 4
        batch_size = observations.shape[0]
        time_step = observations.shape[1]

        image = observations
        image = image.reshape((-1,) + observations.shape[2:])
        image = self._image_colors_to_onehot(image)
        image = image.reshape(batch_size, self.num_colors * self.data_format, image.shape[2], image.shape[3])

        features = self.stem(image)
        for stage in self.stages:
            features = stage(features)

        features = nn.functional.adaptive_max_pool2d(features, 1)
        features = features.flatten(1)
        features = features.reshape(batch_size, -1)
        return self.reason(features).squeeze(-1)

    def ce_loss(self, decisions, targets):
        targets = targets.to(dtype=torch.float, device=decisions.device)
        return nn.functional.binary_cross_entropy_with_logits(
            decisions, targets)

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=self.embed_weights.device),
            self.embed_weights)
        onehot = onehot.permute(0, 3, 1, 2).contiguous()
        return onehot
