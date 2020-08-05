import cv2
import torch
import random
import pickle
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from neuralphys.utils.misc import tprint
from neuralphys.utils.bbox import nonlinear_transform
# import global variable
from neuralphys.utils.config import _C as C


class PyPhys(Dataset):
    def __init__(
        self,
        data_root,
        split,
    ):
        self.data_path = data_root
        self.split = split
        # 1. define the length of input and rollout sequences
        self.input_size = C.RPIN.INPUT_SIZE
        self.cons_size = C.RPIN.CONS_SIZE
        self.infer_start = self.input_size - self.cons_size
        self.pred_size = C.RPIN.PRED_SIZE_TRAIN if self.split == 'train' else C.RPIN.PRED_SIZE_TEST
        # because we are predicting the 'next' frame, we need 1 offset
        self.buffer_size = self.input_size + self.pred_size + 1

        # 2. define output annotations
        self.pred_offset = C.RPIN.OFFSET_LOSS_WEIGHT > 0
        self.pred_position = C.RPIN.POSITION_LOSS_WEIGHT > 0

        # 3. define model configs
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH

        # 4. search ground truth images annotations
        # self.video_list = sorted(glob(f'{self.data_path}/{self.split}/*/'))
        self.video_list = sorted(glob(f'{self.data_path}/{self.split}*/*'))
        self.video_list = [v for v in self.video_list if not v.endswith('.pkl')]
        self.anno_list = sorted(glob(f'{self.data_path}/{self.split}*/*[0-9].pkl'))

        if 'phyre' in self.data_path:
            self.search_suffix = '*_rgb.png'
        elif 'sim' in self.data_path:
            self.search_suffix = '*.pkl'
        else:
            self.search_suffix = '*.jpg'

        if C.INPUT.PRELOAD_TO_MEMORY:
            print('loading data from pickle...')
            with open(f'{self.data_path}/{self.split}.pkl', 'rb') as f:
                data = pickle.load(f)
            self.total_img = np.transpose(data['X'], (0, 1, 4, 2, 3))
            self.total_box = np.zeros((data['y'].shape[:3] + (5,)))
            assert len(self.anno_list) > 0
            for anno_idx, anno_name in enumerate(self.anno_list):
                tprint(f'loading progress: {anno_idx}/{len(self.anno_list)}')
                with open(anno_name, 'rb') as f:
                    boxes = pickle.load(f)
                self.total_box[anno_idx] = boxes

        self.video_info = np.zeros((0, 2), dtype=np.int32)
        vid_size = len(self.video_list if not C.INPUT.PRELOAD_TO_MEMORY else self.total_box)

        for idx, video_name in enumerate(self.video_list if not C.INPUT.PRELOAD_TO_MEMORY else self.total_box):
            # this is only for get image mean and std
            tprint(f'loading progress: {idx}/{vid_size}')
            # use each sliding window as input
            if C.INPUT.PRELOAD_TO_MEMORY:
                valid_seq = self.total_box[idx].shape[0] - self.buffer_size + 1
            else:
                image_list = sorted(glob(f'{video_name}/{self.search_suffix}'))
                valid_seq = self.get_valid_seq(image_list)
                if valid_seq <= 0:
                    continue

            cur_video_info = self.get_video_info(valid_seq, idx)
            self.video_info = np.vstack((self.video_info, cur_video_info))

    def __len__(self):
        return self.video_info.shape[0]

    def __getitem__(self, idx):
        # idx is in the range of valid pairs, get the video index and image starting idx from dict
        vid_idx, img_idx = self.video_info[idx, 0], self.video_info[idx, 1]
        # note: only images are pre-loaded here because
        # 1. image read is the only bottleneck
        # 2. the size of bounding box is different, I would like to pre-process it instead of do it here
        if C.INPUT.PRELOAD_TO_MEMORY:
            data = self.total_img[vid_idx, img_idx:img_idx + self.input_size].copy()
            rois = self.total_box[vid_idx, img_idx:img_idx + self.buffer_size, :, 1:].copy()
        else:
            video_name, anno_name = self.video_list[vid_idx], self.anno_list[vid_idx]
            buffer = np.zeros((self.input_size, 3, self.input_height, self.input_width))
            image_list = sorted(glob(f'{video_name}/{self.search_suffix}'))
            image_list = image_list[img_idx:img_idx + self.input_size]
            for idx, image_name in enumerate(image_list):
                if 'pkl' in image_name:
                    with open(image_name, 'rb') as f:
                        im = pickle.load(f) / 255
                else:
                    im = cv2.imread(image_name)
                buffer[idx, ...] = im.transpose(2, 0, 1)
            data = buffer[:self.input_size].copy()
            with open(anno_name, 'rb') as f:
                boxes = pickle.load(f)
            rois = boxes[img_idx:img_idx + self.buffer_size, :, 1:]

        for c in range(3):
            data[:, c] -= C.INPUT.IMAGE_MEAN[c]
            data[:, c] /= C.INPUT.IMAGE_STD[c]

        if C.RPIN.VAE:
            assert not C.INPUT.PRELOAD_TO_MEMORY
            video_name, anno_name = self.video_list[vid_idx], self.anno_list[vid_idx]
            image_name = sorted(glob(f'{video_name}/{self.search_suffix}'))[img_idx + self.buffer_size - 1]
            if 'pkl' in image_name:
                with open(image_name, 'rb') as f:
                    im = pickle.load(f)
            else:
                im = cv2.imread(image_name)
            data_last = [im.transpose(2, 0, 1)[None, :]]
            data_last = np.concatenate(data_last, axis=0).astype(np.float64)
            for c in range(3):
                data_last[:, c] -= C.INPUT.IMAGE_MEAN[c]
                data_last[:, c] /= C.INPUT.IMAGE_STD[c]

        infer_length = self.pred_size + self.cons_size
        # there are three different boxes used in the model:
        # 1. rois used for region feature extraction
        # 2. src_rois and tar_rois, they are used for computing regression targets
        # normalize input box coordinate to input image scale
        # horizontal flip as data augmentation
        if random.random() > 0.5 and self.split == 'train' and C.RPIN.HORIZONTAL_FLIP:
            rois[..., [0, 2]] = self.input_width - rois[..., [2, 0]]
            data = np.ascontiguousarray(data[..., ::-1])

        if random.random() > 0.5 and self.split == 'train' and C.RPIN.VERTICAL_FLIP:
            rois[..., [1, 3]] = self.input_height - rois[..., [3, 1]]
            data = np.ascontiguousarray(data[..., ::-1, :])

        num_objs = rois.shape[1]
        ignore_mask = np.ones(C.RPIN.NUM_OBJS)
        if num_objs < C.RPIN.NUM_OBJS:
            assert 'phyre' in self.data_path
            rois = np.concatenate([rois, rois[:, :C.RPIN.NUM_OBJS - num_objs]], axis=1)
            ignore_mask[num_objs:] = 0

        src_rois = rois[self.infer_start:][:infer_length].copy()
        tar_rois = rois[self.infer_start + 1:][:infer_length].copy()
        labels, dir_labels = nonlinear_transform(src_rois.reshape(-1, 4), tar_rois.reshape(-1, 4), False)
        labels[..., 0] /= self.input_width
        labels[..., 1] /= self.input_height
        labels = labels.reshape(infer_length, -1, 2)

        # convert to torch tensor
        data = torch.from_numpy(data.astype(np.float32))
        rois = torch.from_numpy(rois.astype(np.float32))
        ignore_mask = torch.from_numpy(ignore_mask.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.float32))
        pos_labels = np.zeros(labels.shape, dtype=np.float32)
        pos_labels[..., 0] = np.sum(tar_rois[..., [0, 2]], axis=-1) / 2 / self.input_width
        pos_labels[..., 1] = np.sum(tar_rois[..., [1, 3]], axis=-1) / 2 / self.input_height
        pos_labels = torch.from_numpy(pos_labels.astype(np.float32))
        labels = torch.cat([labels, pos_labels], dim=-1)

        if C.RPIN.VAE:
            data_last = torch.from_numpy(data_last.astype(np.float32))
        else:
            data_last = data

        return data, rois, labels, data_last, ignore_mask

    def get_valid_seq(self, image_list):
        if 'shape-stack' in self.data_path or 'planning' in self.split:
            valid_seq = 1
        else:
            valid_seq = len(image_list) - self.buffer_size + 1
        return valid_seq

    def get_video_info(self, valid_seq, idx):
        cur_video_info = np.zeros((valid_seq, 2), dtype=np.int32)
        cur_video_info[:, 0] = idx
        cur_video_info[:, 1] = np.arange(valid_seq)
        return cur_video_info
