import phyre
import torch
import time
import numpy as np
import cv2
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from rpin.utils.config import _C as C
from rpin.utils.bbox import xyxy2xywh


class PHYREO(Dataset):
    def __init__(self, data_root, split, image_ext='.jpg'):
        self.data_root = data_root
        self.split = split
        self.image_ext = image_ext
        self.input_size = C.RPIN.INPUT_SIZE  # number of input images
        self.pred_size = eval(f'C.RPIN.PRED_SIZE_{"TRAIN" if split == "train" else "TEST"}')
        self.seq_size = self.input_size + self.pred_size
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH

        protocal = C.PHYRE_PROTOCAL
        fold = C.PHYRE_FOLD

        num_pos = 400 if split == 'train' else 100
        num_neg = 1600 if split == 'train' else 400

        eval_setup = f'ball_{protocal}_template'
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold)
        tasks = train_tasks + dev_tasks if split == 'train' else test_tasks
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)

        # all the actions
        cache = phyre.get_default_100k_cache('ball')
        training_data = cache.get_sample(tasks, None)
        # (100000 x 3)
        actions = training_data['actions']
        # (num_tasks x 100000)
        sim_statuses = training_data['simulation_statuses']

        self.simulator = phyre.initialize_simulator(tasks, action_tier)

        self.video_info = np.zeros((0, 4))
        for t_id, t in enumerate(tqdm(tasks)):
            sim_status = sim_statuses[t_id]
            pos_acts = actions[sim_status == 1].copy()
            neg_acts = actions[sim_status == -1].copy()
            np.random.shuffle(pos_acts)
            np.random.shuffle(neg_acts)
            pos_acts = pos_acts[:num_pos]
            neg_acts = neg_acts[:num_neg]
            acts = np.concatenate([pos_acts, neg_acts])
            video_info = np.zeros((acts.shape[0], 4))
            video_info[:, 0] = t_id
            video_info[:, 1:] = acts
            self.video_info = np.concatenate([self.video_info, video_info])

    def __len__(self):
        return self.video_info.shape[0]

    def __getitem__(self, idx):
        task_id, acts = self.video_info[idx, 0], self.video_info[idx, 1:]
        sim = self.simulator.simulate_action(
            int(task_id), acts, stride=60, need_images=True, need_featurized_objects=True
        )
        images = sim.images
        objs_color = sim.featurized_objects.colors
        objs_valid = [('BLACK' not in obj_color) and ('PURPLE' not in obj_color) for obj_color in objs_color]
        objs = sim.featurized_objects.features[:, objs_valid, :]

        num_objs = objs.shape[1]
        boxes = np.zeros((len(images), num_objs, 5))
        masks = np.zeros((len(images), num_objs, C.RPIN.MASK_SIZE, C.RPIN.MASK_SIZE))

        init_image = cv2.resize(images[0], (C.RPIN.INPUT_WIDTH, C.RPIN.INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)

        if objs[0, :, 8:].sum(0).max() == 1:
            simple_parse = True
        else:
            simple_parse = False

        for im_id, (raw_image, obj) in enumerate(zip(images[:self.seq_size], objs[:self.seq_size])):
            im_height = raw_image.shape[0]
            im_width = raw_image.shape[1]

            obj_ids = np.intersect1d(np.unique(raw_image), [1, 2, 3, 5])

            if simple_parse:
                for o_id, raw_obj_id in enumerate(obj_ids):
                    mask = (raw_image == raw_obj_id)
                    mask = mask[::-1]
                    [h, w] = np.where(mask > 0)
                    x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                    masks[im_id, o_id] = cv2.resize(
                        mask[y1:y2 + 1, x1:x2 + 1].astype(np.float32), (C.RPIN.MASK_SIZE, C.RPIN.MASK_SIZE)
                    ) >= 0.5
                    x1 *= (C.RPIN.INPUT_WIDTH - 1) / (im_width - 1)
                    x2 *= (C.RPIN.INPUT_WIDTH - 1) / (im_width - 1)
                    y1 *= (C.RPIN.INPUT_HEIGHT - 1) / (im_height - 1)
                    y2 *= (C.RPIN.INPUT_HEIGHT - 1) / (im_height - 1)
                    boxes[im_id, o_id] = [o_id, x1, y1, x2, y2]
            else:
                for o_id in range(num_objs):
                    mask = phyre.objects_util.featurized_objects_vector_to_raster(obj[[o_id]])
                    mask = mask[::-1]
                    mask = mask > 0
                    [h, w] = np.where(mask > 0)
                    assert len(h) > 0 and len(w) > 0
                    x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                    masks[im_id, o_id] = cv2.resize(
                        mask[y1:y2 + 1, x1:x2 + 1].astype(np.float32), (C.RPIN.MASK_SIZE, C.RPIN.MASK_SIZE)
                    ) >= 0.5

                    x1 *= (C.RPIN.INPUT_WIDTH - 1) / (im_width - 1)
                    x2 *= (C.RPIN.INPUT_WIDTH - 1) / (im_width - 1)
                    y1 *= (C.RPIN.INPUT_HEIGHT - 1) / (im_height - 1)
                    y2 *= (C.RPIN.INPUT_HEIGHT - 1) / (im_height - 1)
                    boxes[im_id, o_id] = [o_id, x1, y1, x2, y2]

        labels = torch.from_numpy(np.array(int(sim.status == 1), dtype=np.float32))
        boxes = boxes[:self.seq_size, :, 1:]
        data = np.array([phyre.observations_to_float_rgb(init_image)], dtype=np.float).transpose((0, 3, 1, 2))
        data_t = data.copy()
        gt_masks = masks[self.input_size:self.seq_size]

        # pad sequence
        boxes = np.concatenate([boxes] + [boxes[[-1]] for _ in range(self.seq_size - boxes.shape[0])], axis=0)
        gt_masks = np.concatenate(
            [gt_masks] + [gt_masks[[-1]] for _ in range(self.pred_size - gt_masks.shape[0])], axis=0
        )

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
        gt_masks = np.concatenate([gt_masks] + [gt_masks[:, :1] for _ in range(C.RPIN.MAX_NUM_OBJS - num_objs)],
                                  axis=1)

        # rois
        rois = boxes[:self.input_size].copy()
        # gt boxes
        gt_boxes = boxes[self.input_size:].copy()
        gt_boxes = xyxy2xywh(gt_boxes.reshape(-1, 4)).reshape((-1, C.RPIN.MAX_NUM_OBJS, 4))
        gt_boxes[..., 0::2] /= self.input_width
        gt_boxes[..., 1::2] /= self.input_height
        gt_boxes = gt_boxes.reshape(self.pred_size, -1, 4)

        data = torch.from_numpy(data.astype(np.float32))
        data_t = torch.from_numpy(data_t.astype(np.float32))
        rois = torch.from_numpy(rois.astype(np.float32))
        gt_boxes = torch.from_numpy(gt_boxes.astype(np.float32))
        gt_masks = torch.from_numpy(gt_masks.astype(np.float32))
        valid = torch.from_numpy(valid.astype(np.float32))
        return data, data_t, rois, gt_boxes, gt_masks, valid, g_idx, labels
