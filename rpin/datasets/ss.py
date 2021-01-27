import cv2
import torch
import pickle
import numpy as np
from glob import glob

from rpin.datasets.phys import Phys
from rpin.utils.misc import tprint
from rpin.utils.config import _C as C


class SS(Phys):
    def __init__(self, data_root, split, image_ext='.jpg'):
        super().__init__(data_root, split, image_ext)

        self.video_list = sorted(glob(f'{self.data_root}/{self.split}/*/'))
        self.anno_list = [v[:-1] + '_boxes.pkl' for v in self.video_list]

        self.video_info = np.zeros((0, 2), dtype=np.int32)
        for idx, video_name in enumerate(self.video_list):
            tprint(f'loading progress: {idx}/{len(self.video_list)}')
            num_im = len(glob(f'{video_name}/*{image_ext}'))
            # In ShapeStack, we only use the sequence starting from the first frame
            num_sw = min(1, num_im - self.seq_size + 1)
            if num_sw <= 0:
                continue
            video_info_t = np.zeros((num_sw, 2), dtype=np.int32)
            video_info_t[:, 0] = idx  # video index
            video_info_t[:, 1] = np.arange(num_sw)  # sliding window index
            self.video_info = np.vstack((self.video_info, video_info_t))

    def _parse_image(self, video_name, vid_idx, img_idx):
        image_list = sorted(glob(f'{video_name}/*{self.image_ext}'))
        image_list = image_list[img_idx:img_idx + self.input_size]
        data = np.array([
            cv2.imread(image_name) for image_name in image_list
        ], dtype=np.float).transpose((0, 3, 1, 2))
        for c in range(3):
            data[:, c] -= C.INPUT.IMAGE_MEAN[c]
            data[:, c] /= C.INPUT.IMAGE_STD[c]

        image_list = [sorted(glob(f'{video_name}/*{self.image_ext}'))[img_idx + self.seq_size - 1]]
        data_t = np.array([
            cv2.imread(image_name) for image_name in image_list
        ], dtype=np.float).transpose((0, 3, 1, 2))
        for c in range(3):
            data_t[:, c] -= C.INPUT.IMAGE_MEAN[c]
            data_t[:, c] /= C.INPUT.IMAGE_STD[c]

        return data, data_t

    def _parse_label(self, anno_name, vid_idx, img_idx):
        with open(anno_name, 'rb') as f:
            boxes = pickle.load(f)[img_idx:img_idx + self.seq_size, :, 1:]
        gt_masks = np.zeros((self.pred_size, boxes.shape[1], C.RPIN.MASK_SIZE, C.RPIN.MASK_SIZE))
        if C.RPIN.MASK_LOSS_WEIGHT > 0:
            anno_name = anno_name.replace('boxes.', 'masks.')
            with open(anno_name, 'rb') as f:
                gt_masks = pickle.load(f)
            gt_masks = gt_masks[img_idx:img_idx + self.seq_size].astype(np.float32)
            gt_masks = gt_masks[self.input_size:]

        return boxes, gt_masks
