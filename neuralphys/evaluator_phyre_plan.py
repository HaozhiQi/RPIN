import cv2
import torch
import phyre
import random
import numpy as np
from neuralphys.utils.config import _C as C
from neuralphys.utils.misc import tprint, pprint
from neuralphys.utils.bbox import xyxy_to_posf, xyxy_to_rois, xcyc_to_xyxy


class PhyrePlanEvaluator(object):
    def __init__(self, device, pred_model, num_gpus, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        # nn
        self.pred_model = pred_model
        # input setting
        self.input_size = C.RPIN.INPUT_SIZE
        self.cons_size = C.RPIN.CONS_SIZE
        self.pred_size_train, self.pred_size_test = C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        # prediction length
        self.pred_rollout = 40
        self.random_policy = False
        self.pred_model.eval()
        self.ball_radius = 2

    def test(self):
        input_w = input_h = 128
        # PHYRE setup:
        random.seed(0)
        fold_id = 0  # For simplicity, we will just use one fold for evaluation.

        # filter task
        if 'data/phyrec' == C.DATA_ROOT:
            eval_setup = 'ball_cross_template'
            candidate_list = ['00014', '00015', '00019']
        elif 'data/phyre' == C.DATA_ROOT:
            eval_setup = 'ball_within_template'
            candidate_list = ['00000', '00001', '00002', '00007', '00011', '00012', '00013',
                              '00014', '00015', '00016', '00019', '00020', '00024']
        else:
            raise NotImplementedError

        _, _, test_tasks = phyre.get_fold(eval_setup, fold_id)

        test_list = []
        for task in test_tasks:
            if task.split(':')[0] in candidate_list:
                test_list.append(task)

        action_tier = phyre.eval_setup_to_action_tier(eval_setup)
        result_dict = {k: [0, 0, 0] for k in candidate_list}
        cnt = 0
        batched_pred = 200
        all_data, all_rois, all_image = [], [], []
        for idx, task in enumerate(test_list):
            cnt += 1
            simulator = phyre.initialize_simulator([task], action_tier)
            acts = self.enumerate_actions()
            confs = []
            successes = []
            for id_act, act in enumerate(acts):
                if len(successes) >= 2000:
                    break
                tprint(f'eval progress: {cnt} / {len(test_list)} - {id_act} / {len(acts)}')
                # oracle planning
                status, images = simulator.simulate_single(0, act, stride=20, need_images=True)
                if status == phyre.SimulationStatus.INVALID_INPUT:
                    continue
                success = status == phyre.SimulationStatus.SOLVED
                successes.append(success)
                # use random policy
                # continue
                if images is None:
                    continue
                num_objs = len(np.intersect1d(np.unique(images[0]), [1, 2, 3, 5]))
                input_time_step = C.RPIN.INPUT_SIZE
                bbox = np.zeros((len(images[:input_time_step]), num_objs, 5))
                data = []
                for i, image in enumerate(images[:input_time_step]):
                    img = phyre.observations_to_float_rgb(image)
                    image = image[::-1]
                    if i == 0:
                        all_image.append(image)
                    im_height = img.shape[0]
                    im_width = img.shape[1]
                    resize_h, resize_w = input_h / im_height, input_w / im_width
                    img = cv2.resize(img, None, fx=resize_w, fy=resize_h, interpolation=cv2.INTER_LINEAR)
                    data.append(img)
                    o_id = 0
                    # 1 is red ball | 2 is green ball | 3 is blue | 4 is purple | 5 is gray
                    for pixel_id in [1, 2, 3, 5]:
                        [h, w] = np.where(image == pixel_id)
                        if len(h) > 0 and len(w) > 0:
                            x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
                            x1 *= (input_w - 1) / (im_width - 1)
                            x2 *= (input_w - 1) / (im_width - 1)
                            y1 *= (input_h - 1) / (im_height - 1)
                            y2 *= (input_h - 1) / (im_height - 1)
                            bbox[i, o_id] = [o_id, x1, y1, x2, y2]
                            o_id += 1

                data = np.array(data) * 255
                for c in range(3):
                    data[..., c] -= C.INPUT.IMAGE_MEAN[c]
                    data[..., c] /= C.INPUT.IMAGE_STD[c]
                data = data.transpose((0, 3, 1, 2))[None, :]
                data = torch.from_numpy(data.astype(np.float32))
                rois = torch.from_numpy(bbox[..., 1:].astype(np.float32))[None, :]

                all_data.append(data)
                all_rois.append(rois)
                if len(all_data) % batched_pred == 0:
                    all_data = torch.cat(all_data)
                    all_rois = torch.cat(all_rois)
                    traj_array = self.generate_trajs(all_data, all_rois)
                    for i in range(batched_pred):
                        conf = self.get_act_conf(traj_array[[i]], task, all_image[i])
                        confs.append(conf)
                    all_data, all_rois, all_image = [], [], []

            top_acc = np.array(successes)[np.argsort(confs)[::-1]]
            key = task.split(':')[0]
            result_dict[key][0] += int(np.sum(top_acc[:1]) > 0)
            result_dict[key][1] += int(np.sum(top_acc[:100]) > 0)
            result_dict[key][2] += 1
            pprint(result_dict)

        top1 = sum([v[0] for k, v in result_dict.items()]) / sum([v[2] for k, v in result_dict.items()])
        top100 = sum([v[1] for k, v in result_dict.items()]) / sum([v[2] for k, v in result_dict.items()])
        pprint(f'Top-1: {top1 * 100:.2f} | Top-100: {top100 * 100:.2f}')

    def get_act_conf(self, traj_array, task, image):
        pred_window = -10
        if '00000' in task:
            conf = -(((traj_array[0, pred_window:, 1, :] - traj_array[0, pred_window:, 2, :]) ** 2).sum(-1)).sum()
        else:
            [h, w] = np.where(image == 4)
            x1, x2, y1, y2 = w.min(), w.max(), h.min(), h.max()
            goal_x, goal_y = (self.input_width / image.shape[1]) * 0.5 * (x1 + x2), \
                             (self.input_height / image.shape[0]) * 0.5 * (y1 + y2)
            end_center_x = 0.5 * (traj_array[0, pred_window:, 1, 0] + traj_array[0, pred_window:, 1, 2])
            end_center_y = 0.5 * (traj_array[0, pred_window:, 1, 1] + traj_array[0, pred_window:, 1, 3])
            if '00001' in task:
                conf = -np.sqrt((end_center_x - goal_x) ** 2).min()
            if ('00002' in task) or ('00012' in task) or ('00013' in task) or ('00015' in task) \
                    or ('00024' in task):
                conf = -np.sqrt((end_center_y - goal_y) ** 2).min()
            if ('00007' in task) or ('00014' in task) or ('00008' in task) or ('00011' in task) \
                    or ('00016' in task) or ('00019' in task) or ('00020' in task):
                conf = -np.sqrt(((end_center_x - goal_x) ** 2 + (end_center_y - goal_y) ** 2)).min()
        return conf

    @staticmethod
    def enumerate_actions():
        np.random.seed(0)
        actions = np.hstack([np.random.uniform(0, 1, 20000)[:, None],
                             np.random.uniform(0, 1, 20000)[:, None],
                             np.random.uniform(0, 1, 20000)[:, None]])
        return actions

    def generate_trajs(self, data, boxes):
        all_pred_rois = np.zeros((data.shape[0], 0, boxes.shape[2], 4))
        with torch.no_grad():
            pos_feat = xyxy_to_posf(boxes, data.shape)
            rois = xyxy_to_rois(boxes, data.shape[0], data.shape[1], self.num_gpus)
            data, rois, pos_feat = data.to(self.device), rois.to(self.device), pos_feat.to(self.device)
            outputs = self.pred_model(data, rois, pos_feat, num_rollouts=self.pred_rollout + 4)
            bbox_rollouts = outputs['bbox'].cpu().numpy()[..., 2:]
            pred_rois = xcyc_to_xyxy(bbox_rollouts, self.input_height, self.input_width, self.ball_radius)
            all_pred_rois = np.concatenate([all_pred_rois, pred_rois], axis=1)

        return all_pred_rois
