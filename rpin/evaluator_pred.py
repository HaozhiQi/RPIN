import phyre
import torch
import numpy as np
from glob import glob
import torch.nn.functional as F
# ---- NeuralPhys Helper Functions
from rpin.utils.config import _C as C
from rpin.utils.im import get_im_data
from rpin.utils.vis import plot_rollouts
from rpin.utils.misc import tprint, pprint
from rpin.utils.bbox import xyxy_to_rois, xywh2xyxy


class PredEvaluator(object):
    def __init__(self, device, val_loader, model, num_gpus, num_plot_image, output_dir):
        # misc
        self.device = device
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.plot_image = num_plot_image
        # data loader
        self.val_loader = val_loader
        # nn
        self.model = model
        # input setting
        self.ptrain_size, self.ptest_size = C.RPIN.PRED_SIZE_TRAIN, C.RPIN.PRED_SIZE_TEST
        self.input_height, self.input_width = C.RPIN.INPUT_HEIGHT, C.RPIN.INPUT_WIDTH
        # loss settings
        self._setup_loss()
        self.high_resolution_plot = True
        self.vae_num_samples = 100

    def test(self):
        self.model.eval()

        if C.RPIN.VAE:
            losses = dict.fromkeys(self.loss_name, 0.0)
            box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
            masks_step_losses = [0.0 for _ in range(self.ptest_size)]

        for batch_idx, (data, _, rois, gt_boxes, gt_masks, valid, g_idx, seq_l) in enumerate(self.val_loader):
            with torch.no_grad():
                data = data.to(self.device)
                rois = xyxy_to_rois(rois, batch=data.shape[0], time_step=data.shape[1], num_devices=self.num_gpus)
                labels = {
                    'boxes': gt_boxes.to(self.device),
                    'masks': gt_masks.to(self.device),
                    'valid': valid.to(self.device),
                    'seq_l': seq_l.to(self.device),
                }
                outputs = self.model(data, rois, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                self.loss(outputs, labels, 'test')
                # VAE multiple runs
                if C.RPIN.VAE:
                    vae_best_mean = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                    losses_t = self.losses.copy()
                    box_p_step_losses_t = self.box_p_step_losses.copy()
                    masks_step_losses_t = self.masks_step_losses.copy()
                    for i in range(99):
                        outputs = self.model(data, rois, num_rollouts=self.ptest_size, g_idx=g_idx, phase='test')
                        self.loss(outputs, labels, 'test')
                        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
                        if mean_loss < vae_best_mean:
                            losses_t = self.losses.copy()
                            box_p_step_losses_t = self.box_p_step_losses.copy()
                            masks_step_losses_t = self.masks_step_losses.copy()
                            vae_best_mean = mean_loss
                        self._init_loss()

                    for k, v in losses.items():
                        losses[k] += losses_t[k]
                    for i in range(len(box_p_step_losses)):
                        box_p_step_losses[i] += box_p_step_losses_t[i]
                        masks_step_losses[i] += masks_step_losses_t[i]

                tprint(f'eval: {batch_idx}/{len(self.val_loader)}:' + ' ' * 20)

            if self.plot_image > 0:
                outputs = {
                    'boxes': outputs['boxes'].cpu().numpy(),
                    'masks': outputs['masks'].cpu().numpy() if C.RPIN.MASK_LOSS_WEIGHT else None,
                }
                outputs['boxes'][..., 0::2] *= self.input_width
                outputs['boxes'][..., 1::2] *= self.input_height
                outputs['boxes'] = xywh2xyxy(
                    outputs['boxes'].reshape(-1, 4)
                ).reshape((data.shape[0], -1, C.RPIN.MAX_NUM_OBJS, 4))

                labels = {
                    'boxes': labels['boxes'].cpu().numpy(),
                    'masks': labels['masks'].cpu().numpy(),
                }
                labels['boxes'][..., 0::2] *= self.input_width
                labels['boxes'][..., 1::2] *= self.input_height
                labels['boxes'] = xywh2xyxy(
                    labels['boxes'].reshape(-1, 4)
                ).reshape((data.shape[0], -1, C.RPIN.MAX_NUM_OBJS, 4))

                for i in range(rois.shape[0]):
                    batch_size = C.SOLVER.BATCH_SIZE if not C.RPIN.VAE else 1
                    plot_image_idx = batch_size * batch_idx + i
                    if plot_image_idx < self.plot_image:
                        tprint(f'plotting: {plot_image_idx}' + ' ' * 20)
                        video_idx, img_idx = self.val_loader.dataset.video_info[plot_image_idx]
                        video_name = self.val_loader.dataset.video_list[video_idx]

                        v = valid[i].numpy().astype(np.bool)
                        pred_boxes_i = outputs['boxes'][i][:, v]
                        gt_boxes_i = labels['boxes'][i][:, v]

                        if 'PHYRE' in C.DATA_ROOT:
                            im_data = phyre.observations_to_float_rgb(np.load(video_name).astype(np.uint8))[..., ::-1]
                            a, b, c = video_name.split('/')[-3:]
                            output_name = f'{a}_{b}_{c.replace(".npy", "")}'

                            bg_image = np.load(video_name).astype(np.uint8)
                            for fg_id in [1, 2, 3, 5]:
                                bg_image[bg_image == fg_id] = 0
                            bg_image = phyre.observations_to_float_rgb(bg_image)
                        else:
                            bg_image = None
                            image_list = sorted(glob(f'{video_name}/*{self.val_loader.dataset.image_ext}'))
                            im_name = image_list[img_idx]
                            output_name = '_'.join(im_name.split('.')[0].split('/')[-2:])
                            # deal with image data here
                            gt_boxes_i = labels['boxes'][i][:, v]
                            im_data = get_im_data(im_name, gt_boxes_i[None, 0:1], C.DATA_ROOT, self.high_resolution_plot)

                        if self.high_resolution_plot:
                            scale_w = im_data.shape[1] / self.input_width
                            scale_h = im_data.shape[0] / self.input_height
                            pred_boxes_i[..., [0, 2]] *= scale_w
                            pred_boxes_i[..., [1, 3]] *= scale_h
                            gt_boxes_i[..., [0, 2]] *= scale_w
                            gt_boxes_i[..., [1, 3]] *= scale_h

                        pred_masks_i = None
                        if C.RPIN.MASK_LOSS_WEIGHT:
                            pred_masks_i = outputs['masks'][i][:, v]

                        plot_rollouts(im_data, pred_boxes_i, gt_boxes_i,
                                      pred_masks_i, labels['masks'][i][:, v],
                                      output_dir=self.output_dir, output_name=output_name, bg_image=bg_image)

        if C.RPIN.VAE:
            self.losses = losses.copy()
            self.box_p_step_losses = box_p_step_losses.copy()
            self.loss_cnt = len(self.val_loader)

        print('\r', end='')
        print_msg = ""
        mean_loss = np.mean(np.array(self.box_p_step_losses[:self.ptest_size]) / self.loss_cnt) * 1e3
        print_msg += f"{mean_loss:.3f} | "
        print_msg += f" | ".join(["{:.3f}".format(self.losses[name] * 1e3 / self.loss_cnt) for name in self.loss_name])
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
            print_msg += f" | {self.fg_correct / (self.fg_num + 1e-9):.3f} | {self.bg_correct / (self.bg_num + 1e-9):.3f}"
        pprint(print_msg)

    def loss(self, outputs, labels, phase):
        self.loss_cnt += labels['boxes'].shape[0]
        pred_size = eval(f'self.p{phase}_size')
        # calculate bbox loss
        # of shape (batch, time, #obj, 4)
        loss = (outputs['boxes'] - labels['boxes']) ** 2
        # take weighted sum over axis 2 (objs dim) since some index are not valid
        valid = labels['valid'][:, None, :, None]
        loss = loss * valid
        loss = loss.sum(2) / valid.sum(2)
        loss *= self.position_loss_weight

        for i in range(pred_size):
            self.box_p_step_losses[i] += loss[:, i, :2].sum().item()
            self.box_s_step_losses[i] += loss[:, i, 2:].sum().item()

        self.losses['p_1'] = float(np.mean(self.box_p_step_losses[:self.ptrain_size]))
        self.losses['p_2'] = float(np.mean(self.box_p_step_losses[self.ptrain_size:])
                                   if self.ptrain_size < self.ptest_size else 0)
        self.losses['s_1'] = float(np.mean(self.box_s_step_losses[:self.ptrain_size]))
        self.losses['s_2'] = float(np.mean(self.box_s_step_losses[self.ptrain_size:])
                                   if self.ptrain_size < self.ptest_size else 0)

        if C.RPIN.MASK_LOSS_WEIGHT > 0:
            # of shape (batch, time, #obj, m_sz, m_sz)
            mask_loss_ = F.binary_cross_entropy(outputs['masks'], labels['masks'], reduction='none')
            mask_loss = mask_loss_.mean((3, 4))
            valid = labels['valid'][:, None, :]
            mask_loss = mask_loss * valid
            mask_loss = mask_loss.sum(2) / valid.sum(2)
            for i in range(pred_size):
                self.masks_step_losses[i] += mask_loss[:, i].sum().item()

            m1_loss = self.masks_step_losses[:self.ptrain_size]
            m2_loss = self.masks_step_losses[self.ptrain_size:] if self.ptrain_size < self.ptest_size else 0
            self.losses['m_1'] = np.mean(m1_loss)
            self.losses['m_2'] = np.mean(m2_loss)

        if C.RPIN.SEQ_CLS_LOSS_WEIGHT > 0:
            seq_loss = F.binary_cross_entropy(outputs['score'], labels['seq_l'], reduction='none')
            self.losses['seq'] += seq_loss.sum().item()
            # calculate accuracy
            s = (outputs['score'] >= 0.5).eq(labels['seq_l'])
            fg_correct = s[labels['seq_l'] == 1].sum().item()
            bg_correct = s[labels['seq_l'] == 0].sum().item()
            fg_num = (labels['seq_l'] == 1).sum().item()
            bg_num = (labels['seq_l'] == 0).sum().item()
            self.fg_correct += fg_correct
            self.bg_correct += bg_correct
            self.fg_num += fg_num
            self.bg_num += bg_num

        return

    def _setup_loss(self):
        self.loss_name = []
        self.position_loss_weight = C.RPIN.POSITION_LOSS_WEIGHT
        self.loss_name += ['p_1', 'p_2', 's_1', 's_2']
        if C.RPIN.MASK_LOSS_WEIGHT:
            self.loss_name += ['m_1', 'm_2']
        if C.RPIN.SEQ_CLS_LOSS_WEIGHT:
            self.loss_name += ['seq']
        self._init_loss()

    def _init_loss(self):
        self.losses = dict.fromkeys(self.loss_name, 0.0)
        self.box_p_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.box_s_step_losses = [0.0 for _ in range(self.ptest_size)]
        self.masks_step_losses = [0.0 for _ in range(self.ptest_size)]
        # an statistics of each validation
        self.fg_correct, self.bg_correct, self.fg_num, self.bg_num = 0, 0, 0, 0
        self.loss_cnt = 0
