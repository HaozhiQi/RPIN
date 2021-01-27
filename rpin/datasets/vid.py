import torch
import hickle
import random
import phyre
import numpy as np

from glob import glob
from torch.utils.data import Dataset


class VidPHYRECls(Dataset):
    def __init__(self, data_root, split, rs=False, rss=0, rse=0):
        self.data_root = data_root
        self.split = split
        self.pos_dir = f'{self.data_root}/{self.split}/pos/'
        self.neg_dir = f'{self.data_root}/{self.split}/neg/'
        self.pos_data_list = sorted(glob(f'{self.pos_dir}/*/*.hkl'))
        self.neg_data_list = sorted(glob(f'{self.neg_dir}/*/*.hkl'))

    def __len__(self):
        if self.split == 'train':
            # during training, we sample in a balanced way
            return len(self.pos_data_list)
        if self.split == 'test':
            return len(self.pos_data_list) + len(self.neg_data_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            pos_data = hickle.load(self.pos_data_list[idx])
            neg_data = hickle.load(random.choice(self.neg_data_list))

            pos_gt = [pos_data['gt_im'][i] for i in range(min(len(pos_data['gt_im']), 11))] + \
                     [pos_data['gt_im'][-1] for _ in range(11 - len(pos_data['gt_im']))]
            pos_pred = [pos_data['gt_im'][0]] + [pos_data['pred_im'][i] for i in range(10)]

            neg_gt = [neg_data['gt_im'][i] for i in range(min(len(neg_data['gt_im']), 11))] + \
                     [neg_data['gt_im'][-1] for _ in range(11 - len(neg_data['gt_im']))]
            neg_pred = [neg_data['gt_im'][0]] + [neg_data['pred_im'][i] for i in range(10)]

            pos_gt = torch.from_numpy(np.array(pos_gt))
            pos_pred = torch.from_numpy(np.array(pos_pred))
            neg_gt = torch.from_numpy(np.array(neg_gt))
            neg_pred = torch.from_numpy(np.array(neg_pred))

            return pos_gt, pos_pred, neg_gt, neg_pred

        if self.split == 'test':
            if idx < len(self.pos_data_list):
                data = hickle.load(self.pos_data_list[idx])
            else:
                data = hickle.load(self.neg_data_list[idx - len(self.pos_data_list)])

            gt = [data['gt_im'][i] for i in range(min(len(data['gt_im']), 11))] + \
                 [data['gt_im'][-1] for _ in range(11 - len(data['gt_im']))]
            pred = [data['gt_im'][0]] + [data['pred_im'][i] for i in range(10)]
            gt = torch.from_numpy(np.array(gt))
            pred = torch.from_numpy(np.array(pred))
            return gt, pred, torch.ones(1) if idx < len(self.pos_data_list) else torch.zeros(1)