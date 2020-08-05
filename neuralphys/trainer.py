import os
import torch
import numpy as np
from neuralphys.utils.misc import tprint
from timeit import default_timer as timer
from neuralphys.utils.config import _C as C


class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim,
                 max_iters, num_gpus, logger, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.num_gpus = num_gpus
        # data loading
        self.train_loader, self.val_loader = train_loader, val_loader
        # nn optimization
        self.model = model
        self.optim = optim
        # input setting
        self.input_size, self.cons_size = C.RPIN.INPUT_SIZE, C.RPIN.CONS_SIZE
        self.ptrain_size, self.ptest_size = C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        self.batch_size = C.SOLVER.BATCH_SIZE
        # train loop settings
        self.iterations = 0
        self.epochs = 0
        self.max_iters = max_iters
        self.val_interval = C.SOLVER.VAL_INTERVAL
        # loss settings
        self._setup_loss()
        self.best_mean = 1e6

    def train(self):
        print_msg = "| ".join(["progress  | mean "] + list(map("{:6}".format, self.loss_name)))
        self.model.train()
        print('\r', end='')
        self.logger.info(print_msg)
        while self.iterations < self.max_iters:
            self.train_epoch()
            self.epochs += 1

    def train_epoch(self):
        for batch_idx, (data, boxes, labels, data_last, ignore_idx) in enumerate(self.train_loader):
            self._adjust_learning_rate()

            data = data.to(self.device)
            labels = labels.to(self.device)
            rois, coor_features = self._init_rois(boxes, data.shape)
            self.optim.zero_grad()
            outputs = self.model(data, rois, coor_features, num_rollouts=self.ptrain_size + self.cons_size,
                                 data_pred=data_last, phase='train', ignore_idx=ignore_idx)
            loss = self.loss(outputs, labels, 'train', ignore_idx)
            loss.backward()
            self.optim.step()

            self.iterations += self.batch_size

            print_msg = ""
            print_msg += f"{self.epochs:03}/{self.iterations // 1000:04}k"
            print_msg += f" | "
            mean_loss = np.mean(np.array(
                self.pos_step_losses[self.cons_size - 1:self.cons_size + self.ptrain_size]
            ) / self.loss_cnt) * 1e3
            print_msg += f"{mean_loss:.3f} | "
            print_msg += f" | ".join(
                ["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
            speed = self.loss_cnt / (timer() - self.time)
            eta = (self.max_iters - self.iterations) / speed / 3600
            print_msg += f" | speed: {speed:.1f} | eta: {eta:.2f} h"
            print_msg += (" " * (os.get_terminal_size().columns - len(print_msg) - 10))
            tprint(print_msg)

            if self.iterations % self.val_interval == 0:
                self.snapshot()
                self.val()
                self._init_loss()
                self.model.train()

            if self.iterations >= self.max_iters:
                print('\r', end='')
                print(f'{self.best_mean:.3f}')
                break

    def val(self):
        self.model.eval()
        self._init_loss()
        if C.RPIN.VAE:
            all_losses = dict.fromkeys(self.loss_name, 0.0)
            all_step_losses = [0.0 for _ in range(self.ptest_size + self.cons_size)]

        for batch_idx, (data, boxes, labels, _, ignore_idx) in enumerate(self.val_loader):
            tprint(f'eval: {batch_idx}/{len(self.val_loader)}')
            with torch.no_grad():
                data = data.to(self.device)
                labels = labels.to(self.device)
                rois, coor_features = self._init_rois(boxes, data.shape)

                if C.RPIN.VAE:
                    min_losses = dict.fromkeys(self.loss_name, 0.0)
                    min_step_losses = [0.0 for _ in range(self.ptest_size + self.cons_size)]
                    iter_best_mean = 1e6
                    for i in range(10):
                        outputs = self.model(data, rois, coor_features, num_rollouts=self.ptest_size + self.cons_size,
                                             phase='val', ignore_idx=ignore_idx)
                        self.loss(outputs, labels, 'val', ignore_idx)
                        if np.mean(np.array(self.pos_step_losses) / self.loss_cnt) * 1e3 < iter_best_mean:
                            min_losses = self.losses.copy()
                            min_step_losses = self.pos_step_losses.copy()
                            iter_best_mean = np.mean(np.array(self.pos_step_losses)) * 1e3
                        self._init_loss()
                    for k, v in all_losses.items():
                        all_losses[k] += min_losses[k]
                    for i in range(len(all_step_losses)):
                        all_step_losses[i] += min_step_losses[i]
                else:
                    outputs = self.model(data, rois, coor_features, num_rollouts=self.ptest_size + self.cons_size,
                                         phase='val', ignore_idx=ignore_idx)
                    self.loss(outputs, labels, 'val', ignore_idx)

        if C.RPIN.VAE:
            self.losses = all_losses.copy()
            self.pos_step_losses = all_step_losses.copy()
            self.loss_cnt = len(self.val_loader)

        for name in self.loss_name:
            self.losses[name] = self.losses[name] / self.loss_cnt

        print('\r', end='')
        print_msg = ""
        print_msg += f"{self.epochs:03}/{self.iterations // 1000:04}k"
        print_msg += f" | "

        mean_loss = np.mean(np.array(
            self.pos_step_losses[self.cons_size - 1:self.cons_size + self.ptest_size]
        ) / self.loss_cnt) * 1e3

        if np.mean(np.array(self.pos_step_losses[self.cons_size - 1:]) / self.loss_cnt) * 1e3 < self.best_mean:
            self.snapshot('ckpt_best.path.tar')
            self.best_mean = mean_loss

        print_msg += f"{mean_loss:.3f} | "
        print_msg += f" | ".join(["{:.3f}".format(self.losses[name] * 1e3) for name in self.loss_name])
        print_msg += (" " * (os.get_terminal_size().columns - len(print_msg) - 10))
        self.logger.info(print_msg)

    def loss(self, outputs, labels, phase='train', ignore_idx=None):
        self.loss_cnt += labels.shape[0]
        valid_length = self.cons_size + self.ptrain_size if phase == 'train' else self.cons_size + self.ptest_size

        bbox_rollouts = outputs['bbox']
        # of shape (batch, time, #obj, 4)
        loss = (bbox_rollouts - labels) ** 2
        # take mean except time axis, time axis is used for diagnosis
        ignore_idx = ignore_idx[:, None, :, None].to('cuda')
        loss = loss * ignore_idx
        loss = loss.sum(2) / ignore_idx.sum(2)
        loss[..., 0:2] = loss[..., 0:2] * self.offset_loss_weight
        loss[..., 2:4] = loss[..., 2:4] * self.position_loss_weight
        o_loss = loss[..., 0:2]  # offset
        p_loss = loss[..., 2:4]  # position

        for i in range(valid_length):
            self.pos_step_losses[i] += p_loss[:, i].sum(0).sum(-1).mean().item()
            self.off_step_losses[i] += o_loss[:, i].sum(0).sum(-1).mean().item()

        p1_loss = self.pos_step_losses[self.cons_size - 1:self.cons_size + self.ptrain_size]
        p2_loss = self.pos_step_losses[self.cons_size + self.ptrain_size:]
        self.losses['p_1'] = np.mean(p1_loss)
        self.losses['p_2'] = np.mean(p2_loss)

        o1_loss = self.off_step_losses[self.cons_size - 1:self.cons_size + self.ptrain_size]
        o2_loss = self.off_step_losses[self.cons_size + self.ptrain_size:]
        self.losses['o_1'] = np.mean(o1_loss)
        self.losses['o_2'] = np.mean(o2_loss)

        # no need to do precise batch statistics, just do mean for backward gradient
        loss = loss.mean(0)
        pred_length = loss[self.cons_size:].shape[0]
        init_tau = C.RPIN.DISCOUNT_TAU ** (1 / self.ptrain_size)
        tau = init_tau + (self.iterations / self.max_iters) * (1 - init_tau)
        tau = torch.pow(tau, torch.arange(pred_length, out=torch.FloatTensor()))[:, None]
        tau = torch.cat([torch.ones(self.cons_size, 1), tau], dim=0).to('cuda')
        loss = ((loss * tau) / tau.sum(axis=0, keepdims=True)).sum()

        if C.RPIN.VAE and phase == 'train':
            kl_loss = outputs['kl_loss']
            self.losses['k_l'] += kl_loss.sum().item()
            loss += C.RPIN.VAE_KL_LOSS_WEIGHT * kl_loss.sum()

        return loss

    def snapshot(self, name='ckpt_latest.path.tar'):
        torch.save(
            {
                'arch': self.model.__class__.__name__,
                'model': self.model.state_dict(),
            },
            os.path.join(self.output_dir, name),
        )

    def _init_rois(self, boxes, shape):
        batch, time_step, _, height, width = shape
        # coor features, normalized to [0, 1]
        num_im = batch * time_step
        # noinspection PyArgumentList
        co_f = np.zeros(boxes.shape[:-1] + (2,))
        co_f[..., 0] = torch.mean(boxes[..., [0, 2]], dim=-1).numpy().copy() / width
        co_f[..., 1] = torch.mean(boxes[..., [1, 3]], dim=-1).numpy().copy() / height
        coor_features = torch.from_numpy(co_f.astype(np.float32))
        rois = boxes[:, :time_step]
        batch_rois = np.zeros((num_im, C.RPIN.NUM_OBJS))
        batch_rois[np.arange(num_im), :] = np.arange(num_im).reshape(num_im, 1)
        # noinspection PyArgumentList
        batch_rois = torch.FloatTensor(batch_rois.reshape((batch, time_step, -1, 1)))
        # assert batch % self.num_gpus == 0, 'should divide since we drop last in loader'
        load_list = [batch // self.num_gpus for _ in range(self.num_gpus)]
        extra_loaded_gpus = batch - sum(load_list)
        for i in range(extra_loaded_gpus):
            load_list[i] += 1
        load_list = np.cumsum(load_list)
        for i in range(1, self.num_gpus):
            batch_rois[load_list[i - 1]:load_list[i]] -= (load_list[i - 1] * time_step)
        rois = torch.cat([batch_rois, rois], dim=-1)
        return rois, coor_features

    def _setup_loss(self):
        self.loss_name = []
        self.offset_loss_weight = C.RPIN.OFFSET_LOSS_WEIGHT
        self.position_loss_weight = C.RPIN.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2', 'o_1', 'o_2']
        if C.RPIN.VAE:
            self.loss_name += ['k_l']
        self._init_loss()

    def _init_loss(self):
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.pos_step_losses = [0.0 for _ in range(self.ptest_size + self.cons_size)]
        self.off_step_losses = [0.0 for _ in range(self.ptest_size + self.cons_size)]
        # an statistics of each validation
        self.loss_cnt = 0
        self.time = timer()

    def _adjust_learning_rate(self):
        if self.iterations <= C.SOLVER.WARMUP_ITERS:
            lr = C.SOLVER.BASE_LR * self.iterations / C.SOLVER.WARMUP_ITERS
        else:
            if C.SOLVER.SCHEDULER == 'step':
                lr = C.SOLVER.BASE_LR
                for m_iters in C.SOLVER.LR_MILESTONES:
                    if self.iterations > m_iters:
                        lr *= C.SOLVER.LR_GAMMA
            elif C.SOLVER.SCHEDULER == 'cosine':
                lr = 0.5 * C.SOLVER.BASE_LR * (1 + np.cos(np.pi * self.iterations / self.max_iters))
            else:
                raise NotImplementedError

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
