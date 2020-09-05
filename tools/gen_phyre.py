import os
import cv2
import phyre
import random
import pickle
import functools
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

input_w = 128
input_h = 128


def eval_and_save_single_task(task_list_item, action_tier_name, split, cache_name):
    simulator = phyre.initialize_simulator([task_list_item], action_tier_name)
    max_attempts = 30000
    actions = simulator.build_discrete_action_space(max_actions=max_attempts)
    success_actions = []
    fail_actions = []
    random.shuffle(actions)

    for attempt in range(max_attempts):
        if len(success_actions) >= 30:  # only for saving time
            break
        action = actions[attempt]
        status, _ = simulator.simulate_single(0, action)
        if status == phyre.SimulationStatus.SOLVED:
            success_actions.append(action)
        if status == phyre.SimulationStatus.NOT_SOLVED:
            fail_actions.append(action)

    print(f'{simulator.task_ids[0].replace(":", "_")}: success: {len(success_actions)}, fail: {len(fail_actions)}')
    save_root = f'data/{cache_name}/{split}/'
    # select 20 success and 5 fail actions
    random.shuffle(success_actions)
    random.shuffle(fail_actions)
    # sample 1:1 pos/neg samples
    success_actions = success_actions[:50]
    fail_actions = fail_actions[:len(success_actions)]

    for idx, action in enumerate(success_actions + fail_actions):
        status, images = simulator.simulate_single(0, action, stride=20, need_images=True)
        task_id = simulator.task_ids[0]
        save_dir = f'{save_root}/{task_id.replace(":", "_")}_{idx}/'

        num_objs = len(np.intersect1d(np.unique(images[0]), [1, 2, 3, 5]))
        bbox = np.zeros((len(images), num_objs, 5))

        for i, image in enumerate(images):
            img = phyre.observations_to_float_rgb(image)
            image = image[::-1]
            # save raw data
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f'{i:03d}_raw.png'), image)

            im_height = img.shape[0]
            im_width = img.shape[1]
            resize_h, resize_w = input_h / im_height, input_w / im_width
            img = cv2.resize(img, None, fx=resize_w, fy=resize_h, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_dir, f'{i:03d}_rgb.png'), img * 255)

            # plt.imshow(img)
            o_id = 0
            colors = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
            for pixel_id in [1, 2, 3, 5]:
                [h, w] = np.where(image == pixel_id)
                if len(h) > 0 and len(w) > 0:
                    x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                    x1 *= (input_w - 1) / (im_width - 1)
                    x2 *= (input_w - 1) / (im_width - 1)
                    y1 *= (input_h - 1) / (im_height - 1)
                    y2 *= (input_h - 1) / (im_height - 1)
                    bbox[i, o_id] = [o_id, x1, y1, x2, y2]
                    # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=colors[o_id])
                    # plt.gca().add_patch(rect)
                    o_id += 1
            # plt.show()
            # plt.savefig(os.path.join(save_dir, f'{i:03d}_debug.jpg')), plt.close()

        # save bounding boxes
        with open(os.path.join(save_root, f'{task_id.replace(":", "_")}_{idx}.pkl'), 'wb') as f:
            pickle.dump(bbox, f, pickle.HIGHEST_PROTOCOL)
    # print('finish', simulator.task_ids[0])


if __name__ == '__main__':
    # eval_step defines the actions by the first several characters
    # options are 'ball/two_balls' & 'within template/cross template'
    random.seed(0)
    eval_setup = 'ball_within_template'
    cache_name = 'phyre'

    fold_id = 0  # For simplicity, we will just use one fold for evaluation.
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    # train, dev and test partition the whole data set

    train_tasks = sorted(list(train_tasks + dev_tasks))
    test_tasks = sorted(list(test_tasks))
    candidate_list = ['00000', '00001', '00002', '00007', '00011', '00012', '00013',
                      '00014', '00015', '00016', '00019', '00020', '00024']
    train_list, test_list = [], []
    for task in train_tasks:
        if task.split(':')[0] in candidate_list:
            train_list.append(task)
    for task in test_tasks:
        if task.split(':')[0] in candidate_list:
            test_list.append(task)

    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    print('Action tier for', eval_setup, 'is', action_tier)

    for idx, task in enumerate(tqdm(train_list)):
        eval_and_save_single_task(task, action_tier, 'train', cache_name)
    for idx, task in enumerate(tqdm(test_list)):
        eval_and_save_single_task(task, action_tier, 'test', cache_name)
