import cv2
import torch
import hickle
import pickle
import numpy as np
from glob import glob

from rpin.datasets.phys import Phys
from rpin.utils.misc import tprint
from rpin.utils.config import _C as C


class SimB(Phys):
    def __init__(self, data_root, split, image_ext='.jpg'):
        super().__init__(data_root, split, image_ext)

        self.video_list = sorted(glob(f'{self.data_root}/{self.split}/*/'))
        self.anno_list = [v[:-1] + '.pkl' for v in self.video_list]

        if C.INPUT.PRELOAD_TO_MEMORY:
            print('loading data from hickle file...')
            data = hickle.load(f'{self.data_root}/{self.split}.hkl')
            self.total_img = np.transpose(data['X'], (0, 1, 4, 2, 3))
            self.total_box = np.zeros((data['y'].shape[:3] + (5,)))
            for anno_idx, anno_name in enumerate(self.anno_list):
                tprint(f'loading progress: {anno_idx}/{len(self.anno_list)}')
                with open(anno_name, 'rb') as f:
                    boxes = pickle.load(f)
                self.total_box[anno_idx] = boxes

        self.video_info = np.zeros((0, 2), dtype=np.int32)
        for idx, video_name in enumerate(self.video_list if not C.INPUT.PRELOAD_TO_MEMORY else self.total_box):
            tprint(f'loading progress: {idx}/{len(self.video_list)}')
            if C.INPUT.PRELOAD_TO_MEMORY:
                num_sw = self.total_box[idx].shape[0] - self.seq_size + 1
            else:
                num_im = len(glob(f'{video_name}/*{image_ext}'))
                num_sw = num_im - self.seq_size + 1  # number of sliding windows

            if num_sw <= 0:
                continue
            video_info_t = np.zeros((num_sw, 2), dtype=np.int32)
            video_info_t[:, 0] = idx  # video index
            video_info_t[:, 1] = np.arange(num_sw)  # sliding window index
            self.video_info = np.vstack((self.video_info, video_info_t))

    def _parse_image(self, video_name, vid_idx, img_idx):
        if C.INPUT.PRELOAD_TO_MEMORY:
            data = self.total_img[vid_idx, img_idx:img_idx + self.input_size].copy()
        else:
            image_list = sorted(glob(f'{video_name}/*{self.image_ext}'))
            image_list = image_list[img_idx:img_idx + self.input_size]
            data = np.zeros((self.input_size, 3, self.input_height, self.input_width))
            if self.image_ext == '.jpg':  # RealB Case
                data = np.array([
                    cv2.imread(image_name) for image_name in image_list
                ], dtype=np.float).transpose((0, 3, 1, 2))
            else:
                for idx, image_name in enumerate(image_list):
                    with open(image_name, 'rb') as f:
                        data[idx, ...] = (pickle.load(f) / 255).transpose(2, 0, 1)

            for c in range(3):
                data[:, c] -= C.INPUT.IMAGE_MEAN[c]
                data[:, c] /= C.INPUT.IMAGE_STD[c]

        return data

    def _parse_label(self, anno_name, vid_idx, img_idx):
        if C.INPUT.PRELOAD_TO_MEMORY:
            boxes = self.total_box[vid_idx, img_idx:img_idx + self.seq_size, :, 1:].copy()
        else:
            with open(anno_name, 'rb') as f:
                boxes = pickle.load(f)[img_idx:img_idx + self.seq_size, :, 1:]
        gt_masks = np.zeros((self.pred_size, boxes.shape[1], C.RPIN.MASK_SIZE, C.RPIN.MASK_SIZE))
        return boxes, gt_masks
