import os
import phyre
import torch
import hickle
import numpy as np
from glob import glob

from rpin.datasets.phys import Phys
from rpin.utils.misc import tprint
from rpin.utils.config import _C as C

plot = False


class PHYRE(Phys):
    def __init__(self, data_root, split, image_ext='.jpg'):
        super().__init__(data_root, split, image_ext)

        protocal = C.PHYRE_PROTOCAL
        fold = C.PHYRE_FOLD
        env_list = open(f'{data_root}/splits/{protocal}_{split}_fold_{fold}.txt', 'r').read().split('\n')
        self.video_list = sum([sorted(glob(f'{data_root}/images/{env.replace(":", "/")}/*.npy')) for env in env_list], [])
        self.anno_list = [(v[:-4] + '_boxes.hkl').replace('images', 'labels') for v in self.video_list]

        # just for plot images
        if plot:
            self.video_list = [k for k in self.video_list if int(k.split('/')[-1].split('.')[0]) < 40]
            self.anno_list = [k for k in self.anno_list if int(k.split('/')[-1].split('_')[0]) < 40]
            assert len(self.video_list) == len(self.anno_list)
            self.video_list = self.video_list[::80]
            self.anno_list = self.anno_list[::80]

        # video_info_name = f'for_plot.npy'
        video_info_name = f'{data_root}/{protocal}_{split}_{self.input_size}_{self.pred_size}_fold_{fold}_info.npy'
        if os.path.exists(video_info_name):
            print(f'loading info from: {video_info_name}')
            self.video_info = np.load(video_info_name)
        else:
            self.video_info = np.zeros((0, 2), dtype=np.int32)
            for idx, video_name in enumerate(self.video_list):
                tprint(f'loading progress: {idx}/{len(self.video_list)}')
                num_im = hickle.load(video_name.replace('images', 'labels').replace('.npy', '_boxes.hkl')).shape[0]
                if plot:
                    # we will pad sequence so no check
                    num_sw = 1
                else:
                    assert self.input_size == 1
                    num_sw = min(1, num_im - self.seq_size + 1)

                if num_sw <= 0:
                    continue
                video_info_t = np.zeros((num_sw, 2), dtype=np.int32)
                video_info_t[:, 0] = idx  # video index
                video_info_t[:, 1] = np.arange(num_sw)  # sliding window index
                self.video_info = np.vstack((self.video_info, video_info_t))

            np.save(video_info_name, self.video_info)

    def _parse_image(self, video_name, vid_idx, img_idx):
        data = np.array([phyre.observations_to_float_rgb(np.load(video_name))], dtype=np.float).transpose((0, 3, 1, 2))
        return data

    def _parse_label(self, anno_name, vid_idx, img_idx):
        boxes = hickle.load(anno_name)[img_idx:img_idx + self.seq_size, :, 1:]
        gt_masks = np.zeros((self.pred_size, boxes.shape[1], C.RPIN.MASK_SIZE, C.RPIN.MASK_SIZE))
        if C.RPIN.MASK_LOSS_WEIGHT > 0:
            anno_name = anno_name.replace('boxes.', 'masks.')
            gt_masks = hickle.load(anno_name)
            gt_masks = gt_masks[img_idx:img_idx + self.seq_size].astype(np.float32)
            gt_masks = gt_masks[self.input_size:]

        if plot:
            boxes = np.concatenate([boxes] + [boxes[[-1]] for _ in range(self.seq_size - boxes.shape[0])], axis=0)
            gt_masks = np.concatenate(
                [gt_masks] + [gt_masks[[-1]] for _ in range(self.pred_size - gt_masks.shape[0])], axis=0
            )

        return boxes, gt_masks
