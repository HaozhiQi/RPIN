import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from rpin.utils.config import _C as C
from rpin.utils.bbox import xyxy2xywh

plot = False
debug = False


class Phys(Dataset):
    def __init__(self, data_root, split, image_ext='.jpg'):
        self.data_root = data_root
        self.split = split
        self.image_ext = image_ext
        # 1. define property of input and rollout parameters
        self.input_size = C.RPIN.INPUT_SIZE  # number of input images
        self.pred_size = eval(f'C.RPIN.PRED_SIZE_{"TRAIN" if split == "train" else "TEST"}')
        self.seq_size = self.input_size + self.pred_size
        # 2. define model configs
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        self.video_list, self.anno_list = None, None
        self.video_info = None

    def __len__(self):
        return self.video_info.shape[0]

    def __getitem__(self, idx):
        vid_idx, img_idx = self.video_info[idx, 0], self.video_info[idx, 1]
        video_name, anno_name = self.video_list[vid_idx], self.anno_list[vid_idx]
        if C.RPIN.VAE:
            data, data_t = self._parse_image(video_name, vid_idx, img_idx)
        else:
            data = self._parse_image(video_name, vid_idx, img_idx)
            data_t = data.copy()

        boxes, gt_masks = self._parse_label(anno_name, vid_idx, img_idx)

        # image flip augmentation
        if random.random() > 0.5 and self.split == 'train' and C.RPIN.HORIZONTAL_FLIP:
            boxes[..., [0, 2]] = self.input_width - boxes[..., [2, 0]]
            data = np.ascontiguousarray(data[..., ::-1])
            gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])

        if random.random() > 0.5 and self.split == 'train' and C.RPIN.VERTICAL_FLIP:
            boxes[..., [1, 3]] = self.input_height - boxes[..., [3, 1]]
            data = np.ascontiguousarray(data[..., ::-1, :])
            gt_masks = np.ascontiguousarray(gt_masks[..., ::-1])

        # when the number of objects is fewer than the max number of objects
        num_objs = boxes.shape[1]
        g_idx = []
        for i in range(C.RPIN.MAX_NUM_OBJS):
            for j in range(C.RPIN.MAX_NUM_OBJS):
                if j == i:
                    continue
                g_idx.append([i, j, (i < num_objs) * (j < num_objs)])
        g_idx = np.array(g_idx)

        valid = np.ones(C.RPIN.MAX_NUM_OBJS)
        valid[num_objs:] = 0
        boxes = np.concatenate([boxes] + [boxes[:, :1] for _ in range(C.RPIN.MAX_NUM_OBJS - num_objs)], axis=1)
        gt_masks = np.concatenate([gt_masks] + [gt_masks[:, :1] for _ in range(C.RPIN.MAX_NUM_OBJS - num_objs)], axis=1)

        # rois
        rois = boxes[:self.input_size].copy()
        # gt boxes
        gt_boxes = boxes[self.input_size:].copy()
        gt_boxes = xyxy2xywh(gt_boxes.reshape(-1, 4)).reshape((-1, C.RPIN.MAX_NUM_OBJS, 4))
        gt_boxes[..., 0::2] /= self.input_width
        gt_boxes[..., 1::2] /= self.input_height
        gt_boxes = gt_boxes.reshape(self.pred_size, -1, 4)

        labels = torch.zeros(1)  # a fake variable used to make interface consistent
        data = torch.from_numpy(data.astype(np.float32))
        data_t = torch.from_numpy(data_t.astype(np.float32))
        rois = torch.from_numpy(rois.astype(np.float32))
        gt_boxes = torch.from_numpy(gt_boxes.astype(np.float32))
        gt_masks = torch.from_numpy(gt_masks.astype(np.float32))
        valid = torch.from_numpy(valid.astype(np.float32))

        return data, data_t, rois, gt_boxes, gt_masks, valid, g_idx, labels

    def _parse_image(self, video_name, vid_idx, img_idx):
        raise NotImplementedError

    def _parse_label(self, anno_name, vid_idx, img_idx):
        raise NotImplementedError
