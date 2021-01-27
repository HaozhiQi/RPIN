import torch
import numpy as np


def xyxy_to_rois(boxes, batch, time_step, num_devices):
    # convert input bounding box of format (x1, y1, x2, y2) to network input rois
    # two necessary steps are performed:
    # 1. create batch indexes for roi pooling
    # 2. offset the batch_rois for multi-gpu usage
    if boxes.shape[0] != batch:
        assert boxes.shape[0] == (batch // boxes.shape[2])
    batch, num_objs = boxes.shape[0], boxes.shape[2]
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
    co_f = np.zeros(boxes.shape[:-1] + (4,))
    co_f[..., [0, 2]] = boxes[..., [0, 2]].numpy() / width
    co_f[..., [1, 3]] = boxes[..., [1, 3]].numpy() / height
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


def xyxy2xywh(boxes):
    assert boxes.ndim == 2
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0
    xc = boxes[:, 0] + 0.5 * (w - 1.0)
    yc = boxes[:, 1] + 0.5 * (h - 1.0)
    return np.vstack([xc, yc, w, h]).transpose()


def xywh2xyxy(boxes):
    assert boxes.ndim == 2
    xc, yc = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2], boxes[:, 3]
    x1, x2 = xc - 0.5 * (w - 1.0), xc + 0.5 * (w - 1.0)
    y1, y2 = yc - 0.5 * (h - 1.0), yc + 0.5 * (h - 1.0)
    return np.vstack([x1, y1, x2, y2]).transpose()
