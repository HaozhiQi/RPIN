import os
import cv2
import torch
import itertools
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from neuralphys.utils.config import _C as C
from neuralphys.utils.im import sim_rendering, get_im_data
from neuralphys.utils.misc import tprint, pprint
from neuralphys.utils.bbox import xyxy_to_posf, xyxy_to_rois, xcyc_to_xyxy


class PlanEvaluator(object):
    def __init__(self, device, val_loader, pred_model,
                num_gpus, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        # data loader
        self.val_loader = val_loader
        # nn
        self.pred_model = pred_model
        # input setting
        self.input_size = C.RPIN.INPUT_SIZE
        self.cons_size = C.RPIN.CONS_SIZE
        self.pred_size_test = C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        # simulation environment setting
        self.num_objs = C.RPIN.NUM_OBJS
        self.ball_radius = 2.0  # this is not useful for now, just for compatibility
        self.eval_point, self.pred_rollout = 40, 40

        # ------ simulator configs ------
        # init end config
        self.compromise_interval = 5
        self.sim_rollout_length = self.eval_point + self.compromise_interval
        # two ball hitting config
        self.sim_additional = 0
        self.sim_rollout_length = self.eval_point + self.sim_additional
        self.simulator_friction = 1.0
        # oracle setting
        self.oracle_actor = True

        self.task_name = 'hitting'

    def test(self, task_name=''):
        if self.pred_model is not None:
            self.pred_model.eval()

        pos, num_sample = 0, 0
        # sample actions
        acts = self.enumerate_actions()
        for batch_idx, (data, boxes, labels, _, _) in enumerate(self.val_loader):
            batch_size = data.shape[0]
            num_objs = C.RPIN.NUM_OBJS
            pos_feat = xyxy_to_posf(boxes, data.shape)
            rois = xyxy_to_rois(boxes, batch_size, data.shape[1], self.num_gpus)
            gt_rois = boxes.cpu().numpy().copy()

            pred_acts = np.zeros((batch_size, num_objs, 3))
            conf_acts = -np.inf * np.ones((batch_size,))

            for idx, (act, obj_id) in enumerate(itertools.product(acts, range(num_objs))):
                tprint(f'current batch: {idx} / {acts.shape[0] * num_objs}' + ' ' * 10)
                act_array = torch.zeros((batch_size, num_objs, 3), dtype=torch.float32)
                act_array[:, obj_id, :] = torch.from_numpy(act)
                traj_array = self.generate_trajs(data, rois, pos_feat, act_array)
                conf = self.get_act_conf(traj_array, gt_rois, obj_id)
                pred_acts[conf > conf_acts] = act_array[conf > conf_acts]
                conf_acts[conf > conf_acts] = conf[conf > conf_acts]

            sim_rst, debug_gt_traj_array = self.simulate_action(gt_rois, pred_acts)
            pos += sim_rst.sum()
            num_sample += sim_rst.shape[0]
            pprint(f'{task_name} {batch_idx}/{len(self.val_loader)}: {pos / num_sample:.4f}' + ' ' * 10)
        pprint(f'{task_name}: {pos / num_sample:.4f}' + ' ' * 10)

    @staticmethod
    def enumerate_actions():
        sample_act_theta = np.arange(0, 12) / 12.0 * np.pi * 2
        sample_act_vel = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        act_mesh = np.meshgrid(sample_act_theta, sample_act_vel)
        actions = np.hstack([np.sin(act_mesh[0].reshape(-1, 1)),
                             np.cos(act_mesh[0].reshape(-1, 1)),
                             act_mesh[1].reshape(-1, 1)])
        return actions

    def generate_trajs(self, data, rois, pos_feat, acts):
        all_pred_rois = np.zeros((data.shape[0], 0, acts.shape[1], 4))
        with torch.no_grad():
            data, rois, pos_feat, acts = \
                data.to(self.device), rois.to(self.device), pos_feat.to(self.device), acts.to(self.device)

            sim_len_backup = self.sim_rollout_length
            self.sim_rollout_length = self.act_rollout + 1
            _, pred_rois = self.simulate_action(rois[..., 1:].cpu().numpy(), acts.cpu().numpy(), return_rst=False)
            self.sim_rollout_length = sim_len_backup
            all_pred_rois = np.concatenate([all_pred_rois, pred_rois], axis=1)
            pred_rois = all_pred_rois[:, -self.input_size:].copy()

            data = sim_rendering(pred_rois, self.input_height, self.input_width, self.ball_radius)
            for c in range(3):
                data[..., c] -= C.INPUT.IMAGE_MEAN[c]
                data[..., c] /= C.INPUT.IMAGE_STD[c]
            data = data.permute(0, 1, 4, 2, 3)

            # data is (batch x time_step x 3 x h x w)
            boxes = torch.from_numpy(pred_rois.astype(np.float32))
            pos_feat = xyxy_to_posf(boxes, data.shape)
            rois = xyxy_to_rois(boxes, data.shape[0], data.shape[1], self.num_gpus)

            outputs = self.pred_model(data, rois, pos_feat, num_rollouts=self.pred_rollout + self.cons_size)
            bbox_rollouts = outputs['bbox'].cpu().numpy()[..., 2:]
            pred_rois = xcyc_to_xyxy(bbox_rollouts, self.input_height, self.input_width, self.ball_radius)
            pred_rois = pred_rois[:, -(1 + self.pred_rollout):]
            all_pred_rois = np.concatenate([all_pred_rois, pred_rois], axis=1)

        return all_pred_rois

    def get_act_conf(self, pred, gt, obj_id):
        # this function is to get the confidence
        if 'hitting' in self.task_name:
            end_state = pred[:, :self.eval_point]
            init_state = gt[:, [0]]
            dist = (end_state - init_state) ** 2
            dist = np.delete(dist, obj_id, 2)
            dist = dist.sum(3).min(2).max(1)
            return dist
        elif 'init_end' in self.task_name:
            gt_state = gt[:, [self.eval_point]]
            pred_state = pred[:, self.eval_point - 1 - self.compromise_interval:self.eval_point - 1 + self.compromise_interval]
            dist = (gt_state - pred_state) ** 2
            dist = -dist.sum(axis=3).max(axis=2).min(1)
        else:
            raise NotImplementedError
        return dist

    def simulate_action(self, gt, pred_acts, return_rst=True):
        init_state = gt[:, 0]
        sim_rst = np.zeros((init_state.shape[0],))
        traj = np.zeros((pred_acts.shape[0], self.sim_rollout_length, pred_acts.shape[1], 4))
        for i, (p, a) in enumerate(zip(init_state, pred_acts)):
            v = (a[:, :2] * a[:, 2:]).copy()
            x = np.vstack([(p[:, 1] + p[:, 3]) / 2, (p[:, 0] + p[:, 2]) / 2]).transpose().copy()
            hit, traj[i] = self.simulate_rollout(x, v)
            if return_rst:
                sim_rst[i] = self.simulate_rst(hit, traj[i], gt[i])
        return sim_rst, traj

    def simulate_rollout(self, x, v):
        def new_speeds(m1, m2, v1, v2):
            new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
            new_v1 = new_v2 + (v2 - v1)
            return new_v1, new_v2
        eps = .2
        num_obj = x.shape[0]
        r = self.ball_radius * np.ones((num_obj,))
        size = [self.input_height, self.input_width]
        hit = np.diag([1, 1, 1])

        traj = np.zeros((self.sim_rollout_length, num_obj, 4))

        for t in range(self.sim_rollout_length):
            # debugging output
            for i in range(num_obj):
                traj[t, i, 0] = x[i, 1] - self.ball_radius
                traj[t, i, 1] = x[i, 0] - self.ball_radius
                traj[t, i, 2] = x[i, 1] + self.ball_radius
                traj[t, i, 3] = x[i, 0] + self.ball_radius

            col_judge = np.diag([1, 1, 1])
            for mu in range(int(1 / eps)):
                for i in range(num_obj):
                    x[i] += eps * v[i]
                for i in range(num_obj):
                    for z in range(2):
                        if x[i][z] - r[i] < 0:
                            v[i][z] = abs(v[i][z])  # want positive
                        if x[i][z] + r[i] > size[z] - 1:
                            v[i][z] = -abs(v[i][z])  # want negative

                for i in range(num_obj):
                    for j in range(i):
                        if np.linalg.norm(x[i] - x[j]) < r[i] + r[j] and col_judge[i, j] == 0:
                            # the bouncing off part:
                            w = x[i] - x[j]
                            w = w / np.linalg.norm(w)

                            v_i = np.dot(w.transpose(), v[i])
                            v_j = np.dot(w.transpose(), v[j])

                            new_v_i, new_v_j = new_speeds(1, 1, v_i, v_j)

                            v[i] += w * (new_v_i - v_i)
                            v[j] += w * (new_v_j - v_j)
                            hit[i, j] = 1
                            hit[j, i] = 1
                            col_judge[i, j] = 1
                            col_judge[j, i] = 1

            v *= self.simulator_friction

        return hit, traj

    def simulate_rst(self, hit, traj, gt):
        # depending on the task, the correct or incorrect should be described separately
        if 'hitting' in self.task_name:
            assert self.num_objs == 3
            return hit.sum() >= 7
        elif 'init_end' in self.task_name:
            gt_state = gt[[self.eval_point]]
            pred_state = traj[self.eval_point - self.compromise_interval:self.eval_point + self.compromise_interval]
            dist = (gt_state - pred_state) ** 2
            dist[..., 0] = dist[..., 0] + dist[..., 2]
            dist = np.sqrt(dist.sum(2).mean(1)).min()
            return dist
        else:
            raise NotImplementedError

    def reset_param(self):
        if 'init_end' in self.task_name:
            self.eval_point, self.act_rollout, self.pred_rollout = 40, 3, 40
            self.compromise_interval = 5
            self.sim_rollout_length = self.eval_point + self.compromise_interval
        elif 'hitting' in self.task_name:
            self.eval_point, self.act_rollout, self.pred_rollout = 40, 3, 40
            self.sim_additional = 0
            self.sim_rollout_length = self.eval_point + self.sim_additional
        else:
            raise NotImplementedError
