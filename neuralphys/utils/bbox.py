import torch
import numpy as np


def nonlinear_transform(ex_rois, gt_rois, norm_by_rois=True):
    """
    compute bounding box regression targets from ex_rois to gt_rois
    :param ex_rois: [N, 4]
    :param gt_rois: [N, 4]
    :return: [N, 4]
    """
    assert ex_rois.shape[0] == gt_rois.shape[0], 'inconsistent rois number'

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1.0)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1.0)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    if norm_by_rois:
        targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + 1e-14)
        targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + 1e-14)
    else:
        targets_dx = (gt_ctr_x - ex_ctr_x)
        targets_dy = (gt_ctr_y - ex_ctr_y)

    targets = np.vstack(
        (targets_dx, targets_dy)).transpose()
    dir_targets = np.vstack((gt_ctr_x - ex_ctr_x, gt_ctr_y - ex_ctr_y)).transpose()
    return targets, dir_targets


def nonlinear_pred(boxes, box_deltas, normed_by_rois=True):
    """
    Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    :param boxes: !important [N 4]
    :param box_deltas: [N, 4 * num_classes]
    :return: [N 4 * num_classes]
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]

    if normed_by_rois:
        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    else:
        pred_ctr_x = dx + ctr_x[:, np.newaxis]
        pred_ctr_y = dy + ctr_y[:, np.newaxis]

    pred_w = widths[:, np.newaxis]
    pred_h = heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes


def xyxy_to_rois(boxes, batch, time_step, num_devices):
    # convert input bounding box of format (x1, y1, x2, y2) to network input rois
    # two necessary steps are performed:
    # 1. create batch indexes for roi pooling
    # 2. offset the batch_rois for multi-gpu usage
    num_objs = boxes.shape[2]
    num_im = batch * time_step
    rois = boxes[:, :time_step]
    batch_rois = np.zeros((num_im, num_objs))
    batch_rois[np.arange(num_im), :] = np.arange(num_im).reshape(num_im, 1)
    batch_rois = torch.tensor(batch_rois.reshape((batch, time_step, -1, 1)), dtype=torch.float32)
    load_list = [batch // num_devices for _ in range(num_devices)]
    extra_loaded_gpus = batch - sum(load_list)
    for i in range(extra_loaded_gpus):
        load_list[i] += 1
    load_list = np.cumsum(load_list)
    for i in range(1, num_devices):
        batch_rois[load_list[i - 1]:load_list[i]] -= (load_list[i - 1] * time_step)
    rois = torch.cat([batch_rois, rois], dim=-1)
    return rois


def xyxy_to_posf(boxes, shape):
    # convert input bounding box of format (x1, y1, x2, y2) to position feature input
    height, width = shape[-2:]
    co_f = np.zeros(boxes.shape[:-1] + (2,))
    co_f[..., 0] = torch.mean(boxes[..., [0, 2]], dim=-1).numpy().copy() / width
    co_f[..., 1] = torch.mean(boxes[..., [1, 3]], dim=-1).numpy().copy() / height
    coor_features = torch.from_numpy(co_f.astype(np.float32))
    return coor_features


def xcyc_to_xyxy(boxes, scale_h, scale_w, radius):
    # convert input xc yc (normally output by network)
    # to x1, y1, x2, y2 format, with scale and radius offset
    rois = np.zeros(boxes.shape[:3] + (4,))
    rois[..., 0] = boxes[..., 0] * scale_w - radius
    rois[..., 1] = boxes[..., 1] * scale_h - radius
    rois[..., 2] = boxes[..., 0] * scale_w + radius
    rois[..., 3] = boxes[..., 1] * scale_h + radius
    return rois
