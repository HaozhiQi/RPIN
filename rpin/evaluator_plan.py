import os
import cv2
import torch
import phyre
import hickle
import random
import numpy as np
from tqdm import tqdm

from rpin.utils.config import _C as C
from rpin.utils.bbox import xyxy_to_rois, xywh2xyxy


class PlannerPHYRE(object):
    def __init__(self, device, model, score_model, num_gpus, output_dir):
        self.device = device
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.model = model
        self.score_model = score_model
        self.input_size = C.RPIN.INPUT_SIZE
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        self.pred_rollout = C.RPIN.PRED_SIZE_TEST
        self.model.eval()
        self.rendered_mask_idx = [
            [2, 5, 8],
        ]

    def test(self, start_id=0, end_id=25):
        random.seed(0)
        np.random.seed(0)
        protocal, fold_id = C.PHYRE_PROTOCAL, C.PHYRE_FOLD
        self.score_model.eval()
        print(f'testing using protocal {protocal} and fold {fold_id}')

        # setup the PHYRE evaluation split
        eval_setup = f'ball_{protocal}_template'
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        _, _, test_tasks = phyre.get_fold(eval_setup, fold_id)  # PHYRE setup
        candidate_list = [f'{i:05d}' for i in range(start_id, end_id)]  # filter tasks
        test_list = [task for task in test_tasks if task.split(':')[0] in candidate_list]
        simulator = phyre.initialize_simulator(test_list, action_tier)

        # the action candidates are provided by the author of PHYRE benchmark
        num_actions = 10000
        cache = phyre.get_default_100k_cache('ball')
        acts = cache.action_array[:num_actions]
        training_data = cache.get_sample(test_list, None)

        # some statistics variable when doing the evaluation
        auccess = np.zeros((len(test_list), 100))
        batched_pred = C.SOLVER.BATCH_SIZE
        objs_color = None
        all_data, all_acts, all_rois, all_image = [], [], [], []

        # cache the initial bounding boxes from the simulator
        os.makedirs('cache', exist_ok=True)

        t_list = tqdm(test_list, 'Task')
        for task_id, task in enumerate(t_list):
            sim_statuses = training_data['simulation_statuses'][task_id]
            confs, successes = [], []

            boxes_cache_name = f'cache/{task.replace(":", "_")}.hkl'
            use_cache = os.path.exists(boxes_cache_name)
            all_boxes = hickle.load(boxes_cache_name) if use_cache else []

            valid_act_id = 0
            for act_id, act in enumerate(tqdm(acts, 'Candidate Action', leave=False)):
                sim = simulator.simulate_action(task_id, act, stride=60, need_images=True, need_featurized_objects=True)
                assert sim.status == sim_statuses[act_id], 'sanity check not passed'
                if sim.status == phyre.SimulationStatus.INVALID_INPUT:
                    if act_id == len(acts) - 1 and len(all_data) > 0:  # final action is invalid
                        conf_t = self.batch_score(all_data, all_rois, all_image, objs_color)
                        confs = confs + conf_t
                        all_data, all_acts, all_rois, all_image = [], [], [], []
                    continue
                successes.append(sim.status == phyre.SimulationStatus.SOLVED)

                # parse object, prepare input for network, the logic is the same as tools/gen_phyre.py
                image = cv2.resize(sim.images[0], (self.input_width, self.input_height),
                                   interpolation=cv2.INTER_NEAREST)
                all_image.append(image[::-1])
                image = phyre.observations_to_float_rgb(image)
                objs_color = sim.featurized_objects.colors
                objs_valid = [('BLACK' not in obj_color) and ('PURPLE' not in obj_color) for obj_color in objs_color]
                objs = sim.featurized_objects.features[:, objs_valid, :]
                objs_color = np.array(objs_color)[objs_valid]
                num_objs = objs.shape[1]

                if use_cache:
                    boxes = all_boxes[valid_act_id]
                    valid_act_id += 1
                else:
                    boxes = np.zeros((1, num_objs, 5))
                    for o_id in range(num_objs):
                        mask = phyre.objects_util.featurized_objects_vector_to_raster(objs[0][[o_id]])
                        mask_im = phyre.observations_to_float_rgb(mask)
                        mask_im[mask_im == 1] = 0
                        mask_im = mask_im.sum(-1) > 0

                        [h, w] = np.where(mask_im)
                        x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                        x1 *= (self.input_width - 1) / (phyre.SCENE_WIDTH - 1)
                        x2 *= (self.input_width - 1) / (phyre.SCENE_WIDTH - 1)
                        y1 *= (self.input_height - 1) / (phyre.SCENE_HEIGHT - 1)
                        y2 *= (self.input_height - 1) / (phyre.SCENE_HEIGHT - 1)
                        boxes[0, o_id] = [o_id, x1, y1, x2, y2]
                    all_boxes.append(boxes)

                data = image.transpose((2, 0, 1))[None, None, :]
                data = torch.from_numpy(data.astype(np.float32))
                rois = torch.from_numpy(boxes[..., 1:].astype(np.float32))[None, :]

                all_data.append(data)
                all_rois.append(rois)

                if len(all_data) % batched_pred == 0 or act_id == len(acts) - 1:
                    conf_t = self.batch_score(all_data, all_rois, all_image, objs_color)
                    confs = confs + conf_t
                    all_data, all_rois, all_image = [], [], []

            if not use_cache:
                all_boxes = np.stack(all_boxes)
                hickle.dump(all_boxes, boxes_cache_name, mode='w', compression='gzip')

            info = f'current AUCESS: '
            top_acc = np.array(successes)[np.argsort(confs)[::-1]]
            for i in range(100):
                auccess[task_id, i] = int(np.sum(top_acc[:i + 1]) > 0)
            w = np.array([np.log(k + 1) - np.log(k) for k in range(1, 101)])
            s = auccess[:task_id + 1].sum(0) / auccess[:task_id + 1].shape[0]
            info += f'{np.sum(w * s) / np.sum(w) * 100:.2f}'
            t_list.set_description(info)

    def gen_proposal(self, start_id=0, end_id=25):
        random.seed(0)
        np.random.seed(0)
        protocal = C.PHYRE_PROTOCAL
        fold_id = C.PHYRE_FOLD
        print(f'generate proposal for {protocal} fold {fold_id}')
        max_p_acts, max_n_acts, max_acts = 200, 800, 100000
        self.proposal_dir = f'{self.output_dir.split("/")[-1]}_' \
                            f'p{max_p_acts}n{max_n_acts}a{max_acts // 1000}'
        eval_setup = f'ball_{protocal}_template'
        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
        # filter task
        train_tasks = train_tasks + dev_tasks
        candidate_list = [f'{i:05d}' for i in range(start_id, end_id)]

        for split in ['train', 'test']:
            train_list = [task for task in train_tasks if task.split(':')[0] in candidate_list]
            test_list = [task for task in test_tasks if task.split(':')[0] in candidate_list]
            if len(eval(f'{split}_list')) == 0:
                return

            simulator = phyre.initialize_simulator(eval(f'{split}_list'), action_tier)
            cache = phyre.get_default_100k_cache('ball')
            training_data = cache.get_sample(eval(f'{split}_list'), None)
            actions = cache.action_array[:max_acts]

            final_list = eval(f'{split}_list')
            t_list = tqdm(final_list, 'Task')
            for task_id, task in enumerate(t_list):
                box_cache_name = f'data/PHYRE_proposal/cache/{task.replace(":", "_")}_box.hkl'
                act_cache_name = f'data/PHYRE_proposal/cache/{task.replace(":", "_")}_act.hkl'
                use_cache = os.path.exists(box_cache_name) and os.path.exists(act_cache_name)
                if use_cache:
                    acts = hickle.load(act_cache_name)
                    all_boxes = hickle.load(box_cache_name)
                else:
                    sim_statuses = training_data['simulation_statuses'][task_id]
                    pos_acts = actions[sim_statuses == 1]
                    neg_acts = actions[sim_statuses == -1]
                    np.random.shuffle(pos_acts)
                    np.random.shuffle(neg_acts)
                    pos_acts = pos_acts[:max_p_acts]
                    neg_acts = neg_acts[:max_n_acts]
                    acts = np.concatenate([pos_acts, neg_acts])
                    hickle.dump(acts, act_cache_name, mode='w', compression='gzip')
                    all_boxes = []

                valid_act_id = 0
                for act_id, act in enumerate(tqdm(acts, 'Candidate Action', leave=False)):
                    sim = simulator.simulate_action(task_id, act, stride=60, need_images=True,
                                                    need_featurized_objects=True)
                    if not use_cache:
                        if act_id < len(pos_acts):
                            assert sim.status == phyre.SimulationStatus.SOLVED
                        else:
                            assert sim.status == phyre.SimulationStatus.NOT_SOLVED

                    assert sim.status != phyre.SimulationStatus.INVALID_INPUT
                    raw_images = sim.images

                    rst_images = np.stack([np.ascontiguousarray(
                        cv2.resize(rst_image, (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)[::-1]
                    ) for rst_image in raw_images])

                    # prepare input for network:
                    image = cv2.resize(raw_images[0], (self.input_width, self.input_height), interpolation=cv2.INTER_NEAREST)
                    image = phyre.observations_to_float_rgb(image)
                    # parse object
                    objs_color = sim.featurized_objects.colors
                    objs_valid = [('BLACK' not in obj_color) and ('PURPLE' not in obj_color) for obj_color in objs_color]
                    objs = sim.featurized_objects.features[:, objs_valid, :]
                    objs_color = np.array(objs_color)[objs_valid]
                    num_objs = objs.shape[1]

                    if use_cache:
                        boxes = all_boxes[valid_act_id]
                        valid_act_id += 1
                    else:
                        boxes = np.zeros((1, num_objs, 5))
                        for o_id in range(num_objs):
                            mask = phyre.objects_util.featurized_objects_vector_to_raster(objs[0][[o_id]])
                            mask_im = phyre.observations_to_float_rgb(mask)
                            mask_im[mask_im == 1] = 0
                            mask_im = mask_im.sum(-1) > 0

                            [h, w] = np.where(mask_im)
                            x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                            x1 *= (self.input_width - 1) / (phyre.SCENE_WIDTH - 1)
                            x2 *= (self.input_width - 1) / (phyre.SCENE_WIDTH - 1)
                            y1 *= (self.input_height - 1) / (phyre.SCENE_HEIGHT - 1)
                            y2 *= (self.input_height - 1) / (phyre.SCENE_HEIGHT - 1)
                            boxes[0, o_id] = [o_id, x1, y1, x2, y2]
                        all_boxes.append(boxes)

                    data = image.transpose((2, 0, 1))[None, None, :]
                    data = torch.from_numpy(data.astype(np.float32))
                    rois = torch.from_numpy(boxes[..., 1:].astype(np.float32))[None, :]

                    bg_image = rst_images[0].copy()
                    for fg_id in [1, 2, 3, 5]:
                        bg_image[bg_image == fg_id] = 0
                    boxes, masks = self.generate_trajs(data, rois)
                    rst_masks = np.stack([
                        self.render_mask_to_image(boxes[0, i], masks[0, i], images=bg_image.copy(), color=objs_color)
                            .astype(np.uint8) for i in range(self.pred_rollout)
                    ])

                    output_dir = f'data/PHYRE_proposal/{self.proposal_dir}/{split}/'
                    output_dir = output_dir + 'pos/' if sim.status == phyre.SimulationStatus.SOLVED else output_dir + 'neg/'
                    output_dir = output_dir + f'{task.replace(":", "_")}/'
                    os.makedirs(output_dir, exist_ok=True)
                    rst_dict = {'gt_im': rst_images, 'pred_im': rst_masks}
                    hickle.dump(rst_dict, f'{output_dir}/{act_id}.hkl', mode='w', compression='gzip')

                if not use_cache:
                    all_boxes = np.stack(all_boxes)
                    hickle.dump(all_boxes, box_cache_name, mode='w', compression='gzip')

    def render_mask_to_image(self, boxes, masks, images=None, color=None):
        if images is None:
            images = np.zeros((self.input_height, self.input_width))

        color_dict = {
            'WHITE': 0, 'RED': 1, 'GREEN': 2, 'BLUE': 3, 'PURPLE': 4, 'GRAY': 5, 'BLACK': 6
        }

        for o_id, (box, mask) in enumerate(zip(boxes, masks)):
            assert self.input_width == self.input_height
            box = np.maximum(np.minimum(np.round(box).astype(np.int), self.input_height - 1), 0)
            if box[2] - box[0] + 1 <= 0 or box[3] - box[1] + 1 <= 0:
                continue
            mask = cv2.resize(mask, (box[2] - box[0] + 1, box[3] - box[1] + 1))
            mask = (mask >= 0.5)
            images[box[1]:box[3] + 1, box[0]:box[2] + 1][mask] = color_dict[color[o_id]]
        return images

    @staticmethod
    def enumerate_actions():
        tier = 'ball'
        actions = phyre.get_default_100k_cache(tier).action_array[:10000]
        return actions

    def batch_score(self, all_data, all_rois=None, all_image=None, objs_color=None):
        # ours models
        all_data = torch.cat(all_data)
        all_rois = torch.cat(all_rois)
        boxes, masks = self.generate_trajs(all_data, all_rois)

        confs = 0
        for d_i, rendered_mask_idx in enumerate(self.rendered_mask_idx):
            rendered_masks = [[] for _ in rendered_mask_idx]
            for i in range(boxes.shape[0]):
                bg_image = all_image[i].copy()
                # filter out moving objects
                for fg_id in [1, 2, 3, 5]:
                    bg_image[bg_image == fg_id] = 0

                for j, mask_idx in enumerate(rendered_mask_idx):
                    rendered_masks[j].append(self.render_mask_to_image(
                        boxes[i, mask_idx], masks[i, mask_idx], images=bg_image.copy(),  color=objs_color
                    ).astype(np.uint8))

            if d_i == 0:
                data = [torch.from_numpy(np.stack(all_image))] \
                       + [torch.from_numpy(np.stack(env_mask)) for env_mask in rendered_masks]
            else:
                data = [torch.from_numpy(np.stack(env_mask)) for env_mask in rendered_masks]
            data = torch.cat([d[:, None] for d in data], dim=1)

            with torch.no_grad():
                confs = confs + self.score_model(data).sigmoid().cpu().numpy()
        confs /= len(self.rendered_mask_idx)
        return list(confs)

    def generate_trajs(self, data, boxes):
        with torch.no_grad():
            num_objs = boxes.shape[2]
            g_idx = np.array([[i, j, 1] for i in range(num_objs) for j in range(num_objs) if j != i])
            g_idx = torch.from_numpy(g_idx[None].repeat(data.shape[0], 0))
            rois = xyxy_to_rois(boxes, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
            outputs = self.model(data, rois, num_rollouts=self.pred_rollout, g_idx=g_idx)
            outputs = {
                'boxes': outputs['boxes'].cpu().numpy(),
                'masks': outputs['masks'].cpu().numpy(),
            }
            outputs['boxes'][..., 0::2] *= self.input_width
            outputs['boxes'][..., 1::2] *= self.input_height
            outputs['boxes'] = xywh2xyxy(
                outputs['boxes'].reshape(-1, 4)
            ).reshape((data.shape[0], -1, num_objs, 4))

        return outputs['boxes'], outputs['masks']
