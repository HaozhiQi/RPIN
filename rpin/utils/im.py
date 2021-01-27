"""
Helper function for reading image data
These functions were initially designed for visualizations during evaluation
Support high resolution (raw input) and low resolution (network input)
"""
import cv2
import torch
import pickle
import numpy as np


def get_im_data(im_name, gt_rois, data_root, high_res):
    # get image data interface
    # return image data according to dataset
    if 'simb' in data_root:
        im_data = _get_sim_b_im(im_name, gt_rois, high_res)
    elif 'realb' in data_root:
        im_data = _get_real_b_im(im_name, high_res)
    elif 'PHYRE' in data_root:
        im_data = _get_phyre_im(im_name, high_res)
    elif 'shape-stack' in data_root:
        im_data = _get_ss_im(im_name)
    else:
        raise NotImplementedError
    return im_data


def sim_rendering(gt_rois, high_res_h, high_res_w, radius):
    # pred_rois shape should be (batch, timestep, num_obj, 4)
    num_objs = gt_rois.shape[2]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    def ar(x, y, z):
        return z / 2 + np.arange(x, y, z, dtype='float')

    [x_mesh, y_mesh] = np.meshgrid(ar(0, 1, 1. / high_res_w) * high_res_w,
                                   ar(0, 1, 1. / high_res_h) * high_res_h)

    x_mesh, y_mesh = x_mesh[None, None], y_mesh[None, None]
    x_mesh = np.repeat(np.repeat(x_mesh, gt_rois.shape[0], axis=0), gt_rois.shape[1], axis=1)
    y_mesh = np.repeat(np.repeat(y_mesh, gt_rois.shape[0], axis=0), gt_rois.shape[1], axis=1)
    roi_c_x = (gt_rois[..., 0] + gt_rois[..., 2]) / 2
    roi_c_y = (gt_rois[..., 1] + gt_rois[..., 3]) / 2
    im = torch.zeros((gt_rois.shape[0], gt_rois.shape[1], high_res_h, high_res_w, 3),
                     dtype=torch.float32).to('cuda')
    for k in range(num_objs):
        y_mesh_diff = y_mesh - roi_c_y[..., k][..., None, None]
        x_mesh_diff = x_mesh - roi_c_x[..., k][..., None, None]
        mesh_diff = y_mesh_diff ** 2 + x_mesh_diff ** 2
        mesh_diff = torch.from_numpy(mesh_diff).half()
        mesh_diff = mesh_diff.to('cuda')
        pix = torch.exp(-(mesh_diff / (radius ** 2)) ** 4)
        for c in range(3):
            pix_render = colors[k][c] * pix
            im[..., c] += pix_render
    im[im > 1] = 1
    return im


def _get_sim_b_im(im_name, gt_rois, high_res):
    # for simulated billiard
    # high resolution: generate 512 x 512 image using simulation
    # low resolution: directly read 64 x 64 input image
    if high_res:
        high_res_h = 512
        high_res_w = 512
        gt_rois = gt_rois.copy() * 8
        radius = 8
        im = sim_rendering(gt_rois, high_res_h, high_res_w, radius)
        return im.squeeze().cpu().numpy()
    else:
        if 'jpg' in im_name:
            im = cv2.imread(im_name)
        else:
            with open(im_name, 'rb') as f:
                im = pickle.load(f) / 255
    return im


def _get_real_b_im(im_name, high_res):
    # for real billiard
    # high resolution: read image before resizing
    # low resolution: directly read network input image (normally 192 x 96)
    if high_res:
        im_name = im_name.replace('dynamics', '').replace('realb', 'preytb').replace('train/', '').replace('test/', '')
        im = cv2.imread(im_name)
    else:
        im = cv2.imread(im_name)
    return im


def _get_phyre_im(im_name, high_res):
    # for phyre
    # high resolution: read the raw input originally by phyre
    # low resolution: directly reading rgb input of network
    if high_res:
        import phyre
        im = cv2.imread(im_name.replace('rgb', 'raw'), cv2.IMREAD_UNCHANGED)
        im = phyre.observations_to_float_rgb(im)[::-1]
    else:
        im = cv2.imread(im_name)
    return im


def _get_ss_im(im_name):
    # for shape-stack dataset
    # since no resizing is performed for this dataset
    # low resolution is the same as the high resolution
    im = cv2.imread(im_name)
    return im
