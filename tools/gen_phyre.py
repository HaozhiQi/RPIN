import os
import cv2
import phyre
import random
import hickle
import argparse
import numpy as np
from tqdm import tqdm

input_w = 128
input_h = 128
mask_size = 21
max_p_acts = 100
max_n_acts = 400


def arg_parse():
    parser = argparse.ArgumentParser(description='RPIN parameters')
    parser.add_argument('--data', type=str)
    parser.add_argument('--subb', type=int)
    parser.add_argument('--sube', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--gen_data', action='store_true')
    parser.add_argument('--gen_list', action='store_true')
    return parser.parse_args()


def gen_single_task(task_list_item, action_tier_name, cache_name):
    random.seed(0)
    cache = phyre.get_default_100k_cache('ball')
    training_data = cache.get_sample([task_list_item], None)
    actions = cache.action_array
    sim_statuses = training_data['simulation_statuses'][0]

    simulator = phyre.initialize_simulator([task_list_item], action_tier_name)
    pos_acts = actions[sim_statuses == 1]
    neg_acts = actions[sim_statuses == -1]
    print(f'{simulator.task_ids[0].replace(":", "_")}: success: {len(pos_acts)}, fail: {len(neg_acts)}')

    task_id = simulator.task_ids[0]
    im_save_root = f'{cache_name}/images/{task_id.split(":")[0]}/{task_id.split(":")[1]}'
    fim_save_root = f'{cache_name}/full/{task_id.split(":")[0]}/{task_id.split(":")[1]}'
    bm_save_root = f'{cache_name}/labels/{task_id.split(":")[0]}/{task_id.split(":")[1]}'
    os.makedirs(im_save_root, exist_ok=True)
    os.makedirs(fim_save_root, exist_ok=True)
    os.makedirs(bm_save_root, exist_ok=True)

    np.random.shuffle(pos_acts)
    np.random.shuffle(neg_acts)
    pos_acts = pos_acts[:max_p_acts]
    neg_acts = neg_acts[:max_n_acts]
    acts = np.concatenate([pos_acts, neg_acts])

    for act_id, action in enumerate(tqdm(acts)):
        sim = simulator.simulate_action(0, action, stride=60, need_images=True, need_featurized_objects=True)
        images = sim.images
        assert sim.status != 0
        # filter out static objects
        objs_color = sim.featurized_objects.colors
        objs_valid = [('BLACK' not in obj_color) and ('PURPLE' not in obj_color) for obj_color in objs_color]
        objs = sim.featurized_objects.features[:, objs_valid, :]

        num_objs = objs.shape[1]
        boxes = np.zeros((len(images), num_objs, 5))
        masks = np.zeros((len(images), num_objs, mask_size, mask_size))

        full_images = np.zeros((len(images), input_h, input_w))

        for im_id, (raw_image, obj) in enumerate(zip(images, objs)):
            # image = phyre.observations_to_float_rgb(raw_image)
            im_height = raw_image.shape[0]
            im_width = raw_image.shape[1]
            image = cv2.resize(raw_image, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
            if im_id == 0:
                np.save(f'{im_save_root}/{act_id:03d}.npy', image)
            full_images[im_id] = image
            for o_id in range(num_objs):
                mask = phyre.objects_util.featurized_objects_vector_to_raster(obj[[o_id]])
                mask_im = phyre.observations_to_float_rgb(mask)
                mask_im[mask_im == 1] = 0
                mask_im = mask_im.sum(-1) > 0

                [h, w] = np.where(mask_im)

                assert len(h) > 0 and len(w) > 0
                x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                masks[im_id, o_id] = cv2.resize(
                    mask_im[y1:y2 + 1, x1:x2 + 1].astype(np.float32), (mask_size, mask_size)
                ) >= 0.5

                x1 *= (input_w - 1) / (im_width - 1)
                x2 *= (input_w - 1) / (im_width - 1)
                y1 *= (input_h - 1) / (im_height - 1)
                y2 *= (input_h - 1) / (im_height - 1)
                boxes[im_id, o_id] = [o_id, x1, y1, x2, y2]

            # debugging data generation
            # ---- uncomment below for visualize output
            # # debug box output
            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # for o_id in range(num_objs):
            #     x1, y1, x2, y2 = boxes[t, o_id, 1:]
            #     rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2, color='r')
            #     plt.gca().add_patch(rect)
            # plt.savefig(os.path.join(save_dir, f'{t:03d}_debug.jpg')), plt.close()
            # # debug mask output
            # for o_id in range(num_objs):
            #     mask_im = np.zeros((128, 128))
            #     x1, y1, x2, y2 = boxes[t, o_id, 1:].astype(np.int)
            #     mask = cv2.resize(masks[t, o_id].astype(np.float32), (x2 - x1 + 1, y2 - y1 + 1))
            #     mask_im[y1:y2 + 1, x1:x2 + 1] = mask
            #
            #     plt.imshow(mask_im)
            #     plt.savefig(os.path.join(save_dir, f'{t:03d}_{o_id}_debug.jpg')), plt.close()

        # save bounding boxes
        hickle.dump(full_images, f'{fim_save_root}/{act_id:03d}_image.hkl', mode='w', compression='gzip')
        hickle.dump(int(sim.status == 1), f'{bm_save_root}/{act_id:03d}_label.hkl', mode='w', compression='gzip')
        hickle.dump(boxes, f'{bm_save_root}/{act_id:03d}_boxes.hkl', mode='w', compression='gzip')
        hickle.dump(masks, f'{bm_save_root}/{act_id:03d}_masks.hkl', mode='w', compression='gzip')


if __name__ == '__main__':
    # eval_step defines the actions by the first several characters
    # options are 'ball/two_balls' & 'within template/cross template'
    random.seed(0)
    args = arg_parse()
    # misc functions:
    data_dir = args.data
    os.makedirs(data_dir, exist_ok=True)

    if args.gen_list:
        for setup in ['within', 'cross']:
            eval_setup = f'ball_{setup}_template'
            for fold_id in range(10):
                train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
                train_tasks = sorted(list(train_tasks + dev_tasks))
                test_tasks = sorted(list(test_tasks))
                with open(f'{data_dir}/{setup}_train_fold_{fold_id}.txt', 'w') as f:
                    for t in train_tasks:
                        f.write(t + '\n')
                with open(f'{data_dir}/{setup}_test_fold_{fold_id}.txt', 'w') as f:
                    for t in test_tasks:
                        f.write(t + '\n')

    if args.gen_data:
        # this setting does not matter, since we will use the generated list above to refer images
        eval_setup = 'ball_within_template'
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        print('Action tier for', eval_setup, 'is', action_tier)
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, 0)
        train_tasks = sorted(list(train_tasks + dev_tasks + test_tasks))
        candidate_list = [f'{i:05d}' for i in range(args.subb, args.sube)]
        train_list = [task for task in train_tasks if task.split(':')[0] in candidate_list]
        for idx, task in enumerate(tqdm(train_list)):
            gen_single_task(task, action_tier, data_dir)
