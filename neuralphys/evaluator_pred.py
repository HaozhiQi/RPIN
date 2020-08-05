import torch
import numpy as np
from glob import glob
# ---- NeuralPhys Helper Functions
from neuralphys.utils.bbox import xyxy_to_posf, xyxy_to_rois, xcyc_to_xyxy
from neuralphys.utils.config import _C as C
from neuralphys.utils.misc import tprint, pprint


class PredEvaluator(object):
    def __init__(self, device, val_loader, model, num_gpus, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        # data loader
        self.val_loader = val_loader
        # nn
        self.model = model
        # input setting
        self.cons_size, self.pred_size_test = C.RPIN.CONS_SIZE, C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        # loss settings
        self._init_loss()
        self.ball_radius = 2.0  # this is not useful for now, just for compatibility
        self.high_resolution_plot = True
        # other baselines:
        self.vae_num_samples = 100

    def test(self):
        self.model.eval()
        if C.RPIN.VAE:
            all_step_losses = [0.0 for _ in range(self.pred_size_test + self.cons_size)]

        for batch_idx, (data, boxes, labels, _, ignore_idx) in enumerate(self.val_loader):
            with torch.no_grad():
                batch_size = data.shape[0]
                data, labels = data.to(self.device), labels.to(self.device)
                pos_feat = xyxy_to_posf(boxes, data.shape)
                rois = xyxy_to_rois(boxes, batch_size, data.shape[1], self.num_gpus)

                if C.RPIN.VAE:
                    min_step_losses = [0.0 for _ in range(self.pred_size_test + self.cons_size)]
                    iter_best_mean = 100000
                    torch.manual_seed(0)  # enforce deterministic among different runs
                    for i in range(self.vae_num_samples):
                        outputs = self.model(data, rois, pos_feat, num_rollouts=self.pred_size_test + self.cons_size)
                        bbox_rollouts = outputs['bbox']
                        self.loss(bbox_rollouts, labels, ignore_idx)
                        cur_mean = np.mean(np.array(self.step_losses) / self.loss_cnt) * 1e3
                        if cur_mean < iter_best_mean:
                            min_step_losses = self.step_losses.copy()
                            iter_best_mean = cur_mean
                        self._init_loss()

                    # noinspection PyUnboundLocalVariable
                    for i in range(len(all_step_losses)):
                        all_step_losses[i] += min_step_losses[i]
                    t_eval_info = f'{np.mean(np.array(all_step_losses)) / (batch_idx + 1) * 1000:.3f}'
                    tprint(f'eval: {batch_idx}/{len(self.val_loader)}: ' + t_eval_info + ' ' * 10)
                else:
                    outputs = self.model(data, rois, pos_feat, num_rollouts=self.pred_size_test + self.cons_size,
                                         ignore_idx=ignore_idx)
                    bbox_rollouts = outputs['bbox']
                    bbox_rollouts = torch.clamp(bbox_rollouts, 0, 1)  # avoid overflow
                    self.loss(bbox_rollouts, labels, ignore_idx)
                    tprint(f'eval: {batch_idx}/{len(self.val_loader)}:' + ' ' * 20)

        if C.RPIN.VAE:
            self.step_losses = all_step_losses.copy()
            self.loss_cnt = len(self.val_loader)
        self.step_losses = list(np.array(self.step_losses) / self.loss_cnt)

        step_loss_all = np.array(self.step_losses[self.cons_size - 1:])
        loss_list = [step_loss_all]
        if 'shape-stack' in C.DATA_ROOT:
            loss_list += [step_loss_all[:16], step_loss_all[16:31]]
        else:
            loss_list += [step_loss_all[:21], step_loss_all[21:41]]
        name_list = ['all', '0-16', '16-31'] if 'shape-stack' in C.DATA_ROOT else ['all', '0-20', '21-40']

        print_msg = ''
        for idx, (name, loss) in enumerate(zip(name_list, loss_list)):
            print_msg += name + f': {loss.mean() * 1000:.3f} | '
            if idx > 0:
                for li in loss:
                    print_msg += f"{li * 1000:.2f} "
            if idx != len(name_list) - 1:
                print_msg += '\n'
        pprint(print_msg)

    def loss(self, bbox_rollouts, labels, ignore_idx):
        self.loss_cnt += labels.shape[0]
        # of shape (batch, time, #obj, 4)
        loss = (bbox_rollouts - labels) ** 2
        ignore_idx = ignore_idx[:, None, :, None].to('cuda')
        loss = loss * ignore_idx
        loss = loss.sum(2) / ignore_idx.sum(2)
        loss = loss[..., 2:4]
        # count loss at each time step
        for i in range(self.pred_size_test + self.cons_size):
            self.step_losses[i] += loss[:, i].sum(0).sum(-1).mean().item()
        return

    def _init_loss(self):
        self.step_losses = [0.0 for _ in range(self.pred_size_test + self.cons_size)]
        self.loss_cnt = 0
