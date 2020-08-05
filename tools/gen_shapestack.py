import os
import cv2
import pickle
import shutil
import numpy as np

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt


if __name__ == '__main__':
    cache_name = 'data/shape-stack/test/'
    # cache_name = 'debug'
    r = 35
    os.makedirs(cache_name, exist_ok=True)
    data_path = 'data/shapestacks'
    src_path = os.path.join(data_path, 'splits', 'env_ccs+blocks-hard+easy-h=3-vcom=1+2+3-vpsf=0')
    vid_dir = os.path.join(data_path, 'frc_35')
    datasets = ['eval']
    for dataset in datasets:
        data_list_name = f'{src_path}/{dataset}.txt'
        with open(data_list_name) as f:
            vid_name_list = [line.split()[0] for line in f]
        for vid_id, vid_name in enumerate(tqdm(vid_name_list)):
            vid_path = os.path.join(vid_dir, vid_name)
            image_list = sorted(glob(os.path.join(vid_path, '*.jpg')))
            bboxes = np.load(os.path.join(vid_path, 'cam_1.npy'))
            num_objs = bboxes.shape[1]
            rst_bboxes = np.zeros((bboxes.shape[0], bboxes.shape[1], 5))
            for t in range(bboxes.shape[0]):
                rst_bboxes[t, :, 0] = np.arange(num_objs)
                rst_bboxes[t, :, 1] = bboxes[t, :, 0] * 224 - r
                rst_bboxes[t, :, 2] = bboxes[t, :, 1] * 224 - r
                rst_bboxes[t, :, 3] = bboxes[t, :, 0] * 224 + r
                rst_bboxes[t, :, 4] = bboxes[t, :, 1] * 224 + r
            with open(f'{cache_name}/{vid_id:04d}.pkl', 'wb') as f:
                pickle.dump(rst_bboxes, f, pickle.HIGHEST_PROTOCOL)
