import os
import cv2
import pickle
import imageio
import numpy as np
from matplotlib import pyplot as plt


def _hex_to_ints(hex_string):
    hex_string = hex_string.strip('#')
    return (
        int(hex_string[0:2], 16),
        int(hex_string[2:4], 16),
        int(hex_string[4:6], 16),
    )


WAD_COLORS = np.array(
    [
        [255, 255, 255],  # White.
        _hex_to_ints('f34f46'),  # Red.
        _hex_to_ints('6bcebb'),  # Green.
        _hex_to_ints('1877f2'),  # Blue.
        _hex_to_ints('4b4aa4'),  # Purple.
        _hex_to_ints('b9cad2'),  # Gray.
        [0, 0, 0],  # Black.
        _hex_to_ints('fcdfe3'),  # Light red.
    ],
    dtype=np.uint8)


def plot_rollouts(im_data, pred_boxes, gt_boxes, pred_masks=None, gt_masks=None,
                  output_dir='', output_name='', bg_image=None):
    # plot rollouts for different dataset
    # 1. plot images
    # 2. plot bounding boxes
    # 3. plot masks (optional)
    im_ext = 'png'
    kwargs = {'format': im_ext, 'bbox_inches': 'tight', 'pad_inches': 0}
    bbox_dir = os.path.join(output_dir, 'bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    # # there are several cases:
    # # 1. overlapped gt and predictions
    # plt.axis('off')
    # plt.imshow(im_data[..., ::-1], alpha=0.7)
    # _plot_bbox_traj(pred_rois, size=80, alpha=1.0)
    # _plot_bbox_traj(gt_rois, size=320, alpha=0.6, facecolors='none')
    # plt.savefig(f'{bbox_dir}/ov_{output_name}.{im_ext}', **kwargs)
    # plt.close()
    # 2. side-by-side gt and predictions
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(im_data[..., ::-1])
    _plot_bbox_traj(pred_boxes, size=160, alpha=1.0)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(im_data[..., ::-1])
    _plot_bbox_traj(gt_boxes, size=160, alpha=1.0)
    plt.savefig(f'{bbox_dir}/sbs_{output_name}.{im_ext}', **kwargs)
    plt.close()
    # # 3. separate print two images
    # plt.axis('off')
    # plt.imshow(im_data[..., ::-1])
    # _plot_bbox_traj(pred_rois, size=160, alpha=1.0)
    # plt.savefig(f'{bbox_dir}/pred_{output_name}.{im_ext}', **kwargs)
    # plt.close()
    # plt.axis('off')
    # plt.imshow(im_data[..., ::-1])
    # _plot_bbox_traj(gt_rois, size=160, alpha=1.0)
    # plt.savefig(f'{bbox_dir}/gt_{output_name}.{im_ext}', **kwargs)
    # plt.close()

    # # 4. print videos, need png since ffmpeg cannot deal with svg
    # # disable at default
    # video_dir = os.path.join(output_dir, 'video')
    # os.makedirs(video_dir, exist_ok=True)
    # kwargs['format'] = 'png'
    # for i in range(gt_boxes.shape[0]):
    #     plt.axis('off')
    #     plt.imshow(im_data[..., ::-1])
    #     _plot_bbox_traj(gt_boxes[:i+1], size=320, alpha=1.0)
    #     plt.savefig(f'{video_dir}/gt_{output_name}_{i}.png', **kwargs)
    #     plt.close()
    # for i in range(pred_boxes.shape[0]):
    #     plt.axis('off')
    #     plt.imshow(im_data[..., ::-1])
    #     _plot_bbox_traj(pred_boxes[:i+1], size=320, alpha=1.0)
    #     plt.savefig(f'{video_dir}/pred_{output_name}_{i}.png', **kwargs)
    #     plt.close()

    if pred_masks is None:
        return

    mask_dir = os.path.join(output_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    # currently the following are only for the [shape-stack] dataset
    # the radius is assumed to be 35 (as in original CVP paper)
    mask_colors = [
        [1.0, 0.7, 0.4],
        [0.4, 0.4, 1.0],
        [0.65, 0.65, 0.65],
    ]

    # comment if not use PHYRE

    env_name = output_name.split('_')[0]
    if env_name in ['00000']:
        mask_colors = [WAD_COLORS[2], WAD_COLORS[3], WAD_COLORS[1]]
    elif env_name in ['00004', '00006']:  # BLUE + GREEN
        mask_colors = [WAD_COLORS[3], WAD_COLORS[2], WAD_COLORS[1]]
    elif env_name in ['00001', '00002', '00007', '00008', '00009', '00011', '00012', '00013', '00014', '00015']:
        mask_colors = [WAD_COLORS[2], WAD_COLORS[1]]
    elif env_name in ['00003', '00005', '00010', '00017', '00021']:  # GRAY + GREEN
        mask_colors = [WAD_COLORS[5], WAD_COLORS[2], WAD_COLORS[1]]
    elif env_name in ['00018']:  # 2 GRAY + GREEN + BLUE
        mask_colors = [WAD_COLORS[5], WAD_COLORS[3], WAD_COLORS[5], WAD_COLORS[2], WAD_COLORS[1]]
    elif env_name in ['00022']:  # 1 GRAY + GREEN + BLUE
        mask_colors = [WAD_COLORS[5], WAD_COLORS[3], WAD_COLORS[2], WAD_COLORS[1]]
    elif env_name in ['00023']:  # 1 GRAY + GREEN + BLUE
        mask_colors = [WAD_COLORS[5], WAD_COLORS[5], WAD_COLORS[2], WAD_COLORS[5], WAD_COLORS[3], WAD_COLORS[1]]
    elif env_name in ['00016', '00019', '00020', '00024']:
        mask_colors = [WAD_COLORS[2], WAD_COLORS[5], WAD_COLORS[1]]

    # plt.figure(figsize=(12, 12))
    # plt.subplot(1, 2, 1)
    # plt.imshow(im_data[..., ::-1])
    # plt.subplot(1, 2, 2)
    # plt.imshow(im_data[..., ::-1])
    # plt.savefig(f'{mask_dir}/{output_name}_{0}.{im_ext}', **kwargs)
    # plt.close()

    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(im_data[..., ::-1])
    # plt.savefig(f'{bbox_dir}/pred_{output_name}_0.{im_ext}', **kwargs)
    # plt.close()
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(im_data[..., ::-1])
    # plt.savefig(f'{bbox_dir}/gt_{output_name}_0.{im_ext}', **kwargs)
    # plt.close()

    time_step, num_objs = pred_boxes.shape[:2]
    for t_id in range(time_step):
        gt_mask_im = bg_image.copy() * 255
        pred_mask_im = bg_image.copy() * 255
        for o_id in range(num_objs):
            gt_bbox_t_o = np.maximum(np.minimum(np.round(gt_boxes[t_id, o_id]).astype(np.int), 127), 0)
            gt_mask_t_o = cv2.resize(gt_masks[t_id, o_id], (gt_bbox_t_o[2] - gt_bbox_t_o[0] + 1,
                                                            gt_bbox_t_o[3] - gt_bbox_t_o[1] + 1))
            gt_mask_t_o = (gt_mask_t_o >= 0.5)
            for c_id in range(3):
                gt_mask_im[gt_bbox_t_o[1]:gt_bbox_t_o[3] + 1,
                           gt_bbox_t_o[0]:gt_bbox_t_o[2] + 1, c_id][gt_mask_t_o] = mask_colors[o_id][c_id]

            pred_bbox_t_o = np.maximum(np.minimum(np.round(pred_boxes[t_id, o_id]).astype(np.int), 127), 0)
            pred_mask_t_o = cv2.resize(pred_masks[t_id, o_id], (pred_bbox_t_o[2] - pred_bbox_t_o[0] + 1,
                                                                pred_bbox_t_o[3] - pred_bbox_t_o[1] + 1))
            pred_mask_t_o = (pred_mask_t_o >= 0.5)
            for c_id in range(3):
                pred_mask_im[pred_bbox_t_o[1]:pred_bbox_t_o[3] + 1,
                             pred_bbox_t_o[0]:pred_bbox_t_o[2] + 1, c_id][pred_mask_t_o] = mask_colors[o_id][c_id]

        plt.figure(figsize=(12, 12))
        plt.subplot(1, 2, 1)
        pred_mask_im = np.minimum(np.maximum(pred_mask_im, 0.0), 255.0)
        plt.imshow(pred_mask_im.astype(np.uint8))
        plt.subplot(1, 2, 2)
        gt_mask_im = np.minimum(np.maximum(gt_mask_im, 0.0), 255.0)
        plt.imshow(gt_mask_im.astype(np.uint8))
        plt.savefig(f'{mask_dir}/{output_name}_{t_id + 1}.{im_ext}', **kwargs)
        plt.close()
        # plt.figure(figsize=(12, 12))
        # plt.axis('off')
        # pred_mask_im = np.minimum(np.maximum(pred_mask_im, 0.0), 255.0)
        # plt.imshow(pred_mask_im.astype(np.uint8))
        # plt.savefig(f'{mask_dir}/pred_{output_name}_{t_id + 1}.{im_ext}', **kwargs)
        # plt.close()
        # plt.figure(figsize=(12, 12))
        # plt.axis('off')
        # gt_mask_im = np.minimum(np.maximum(gt_mask_im, 0.0), 255.0)
        # plt.imshow(gt_mask_im.astype(np.uint8))
        # plt.savefig(f'{mask_dir}/gt_{output_name}_{t_id + 1}.{im_ext}', **kwargs)
        # plt.close()

    # r = 35
    # gt_xc = (0.5 * (gt_rois[..., 0] + gt_rois[..., 2]) + 2 * r).astype(np.int)
    # gt_yc = (0.5 * (gt_rois[..., 1] + gt_rois[..., 3]) + 2 * r).astype(np.int)
    # pred_xc = (0.5 * (pred_rois[..., 0] + pred_rois[..., 2]) + 2 * r).astype(np.int)
    # pred_yc = (0.5 * (pred_rois[..., 1] + pred_rois[..., 3]) + 2 * r).astype(np.int)
    #
    # time_step, num_obj, _ = pred_rois.shape
    # for t_id in range(time_step):
    #     gt_mask_im = np.zeros((224 + 4 * r, 224 + 4 * r, 3))
    #     pred_mask_im = np.zeros((224 + 4 * r, 224 + 4 * r, 3))
    #     for o_id in range(num_obj):
    #         gt_mask_t_o = cv2.resize(gt_masks[t_id, o_id], (2 * r, 2 * r)) >= 0.5
    #         xc_t_o, yc_t_o = int(gt_xc[t_id, o_id]), int(gt_yc[t_id, o_id])
    #
    #         if xc_t_o < r or yc_t_o < r or xc_t_o > 224 + 3 * r or yc_t_o > 224 + 3 * r:
    #             continue
    #
    #         for c_id in range(3):
    #             c = mask_colors[o_id][c_id]
    #             gt_mask_im[yc_t_o - r:yc_t_o + r, xc_t_o - r:xc_t_o + r, c_id][gt_mask_t_o] += c
    #
    #         pred_mask_t_o = cv2.resize(pred_masks[t_id, o_id], (2 * r, 2 * r)) >= 0.5
    #         xc_t_o, yc_t_o = int(pred_xc[t_id, o_id]), int(pred_yc[t_id, o_id])
    #
    #         if xc_t_o < r or yc_t_o < r or xc_t_o > 224 + 3 * r or yc_t_o > 224 + 3 * r:
    #             continue
    #
    #         for c_id in range(3):
    #             c = mask_colors[o_id][c_id]
    #             pred_mask_im[yc_t_o - r:yc_t_o + r, xc_t_o - r:xc_t_o + r, c_id][pred_mask_t_o] += c
    #
    #     gt_mask_im = gt_mask_im[2 * r:-2 * r, 2 * r:-2 * r]
    #     pred_mask_im = pred_mask_im[2 * r:-2 * r, 2 * r:-2 * r]
    #     plt.figure(figsize=(12, 12))
    #     plt.subplot(1, 2, 1)
    #     pred_mask_im = np.minimum(np.maximum(pred_mask_im, 0.0), 1.0)
    #     plt.imshow(pred_mask_im)
    #     plt.subplot(1, 2, 2)
    #     gt_mask_im = np.minimum(np.maximum(gt_mask_im, 0.0), 1.0)
    #     plt.imshow(gt_mask_im)
    #     plt.savefig(f'{mask_dir}/{output_name}_{t_id}.svg', format='svg')
    #     plt.close()


def _plot_bbox_traj(bboxes, size=80, alpha=1.0, facecolors=None):
    for idx, bbox in enumerate(bboxes):
        inst_id = 0
        color_progress = idx / len(bboxes)
        color_cyan = (0, 1 - 0.4 * color_progress, 1 - 0.4 * color_progress)
        color_brown = (0.4 + 0.4 * color_progress, 0.2 + 0.2 * color_progress, 0.0)
        color_red = (1.0 - 0.3 * color_progress, 0.0, 0.0)
        color_purple = (0.3 + 0.4 * color_progress, 0.4 * color_progress, 0.6 + 0.4 * color_progress)
        color_orange = (1.0, 0.5 + 0.3 * color_progress, 0.4 * color_progress)
        color = [color_red, color_purple, color_orange, color_cyan, color_cyan, color_brown]
        for obj in bbox:
            # rect = plt.Rectangle((obj[0], obj[1]), obj[2] - obj[0], obj[3] - obj[1],
            #                      linewidth=3, edgecolor=color[inst_id], facecolor='none')
            # plt.gca().add_patch(rect)
            ctr_x = 0.5 * (obj[0] + obj[2]) if len(obj) == 4 else obj[0]
            ctr_y = 0.5 * (obj[1] + obj[3]) if len(obj) == 4 else obj[1]
            plt.scatter(ctr_x, ctr_y, size, alpha=alpha * (1 - 0.4 * color_progress),
                        color=color[inst_id], facecolors=facecolors)
            inst_id += 1


def plot_data(data):
    assert data.ndim == 5
    batch, time_step = data.shape[:2]
    for data_b in data:
        for data_b_t in data_b:
            data_b_t = data_b_t.transpose((1, 2, 0))
            plt.imshow(data_b_t)
            plt.show()
