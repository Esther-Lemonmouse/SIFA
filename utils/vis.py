#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File       vis.py
@Author     Shuxin Yao
@Email      yaosx_job@outlook.com
@CreateTime 2022/07/22
@Function   please provide a brief discription here
"""
import os
import cv2
import numpy as np

MODALITY = 'B2A'

def slice_crop(img_vol, slice_idx):
    if slice_idx >= img_vol.shape[2] or slice_idx < 0:
        raise IndexError('Selected slice index is out of the boundary!')
    return img_vol[..., slice_idx]


def colormap_generator(num_cls, seed):
    return [tuple(seed.integers(256, size=3)) for cls in range(1, num_cls)]


def mask_generator(mask_img, colormap):
    # 转换uint8便于操作
    mask_img = mask_img.astype(np.uint8)
    # 找到类别编号
    cls = mask_img.max()
    # 生成对应BGR三通道图
    mask_img_color = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    # 上色，注意类别要减一
    mask_img_color[mask_img==cls] = colormap[cls-1]
    return mask_img_color


def image_masks_merge(img, mask_list):
    # list转ndarray便于加成一张图
    mask_list = np.array(mask_list)
    mask_image = mask_list.sum(axis=0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    merged_image = (img + 0.4*mask_image).astype(np.uint8)
    return merged_image


def save_img(img, file_pth):
    cv2.imwrite(file_pth, img)


def relate_list_generator(filename):
    # 注意文件命名规则：Domain编号在首位，case编号在末位
    name_parts = filename.strip('.npz').split('_')
    relate_list = [fn for fn in os.listdir()
                   if fn.startswith(name_parts[0]) and name_parts[-1] in fn and fn.endswith('npy')]
    relate_list.sort()
    return relate_list


def main_process(gth=False, class_num=2):
    os.chdir('../visualization/')
    rng = np.random.default_rng(17)
    colormap = colormap_generator(class_num, rng)
    target_img_list = [fn for fn in os.listdir() if fn.endswith('npz')]
    if not target_img_list:
        raise ValueError('Please move the test images manually into directory "visualization". ')

    for target_name in target_img_list:
        target = np.load(target_name)
        target_image = target['arr_0']
        masks_name_list = relate_list_generator(target_name)
        # class_num = len(masks_name_list)+1

        slice_idx = np.floor(target_image.shape[2] / 2).astype(np.int8)
        target_image_norm = slice_crop(target_image, slice_idx)
        target_image = ((target_image_norm+target_image_norm.min())
                        *(255/(target_image_norm.max()-target_image_norm.min()))).astype(np.uint8)
        pred_mask_image_list = []
        if gth:
            target_label = target['arr_1']
            target_label = slice_crop(target_label, slice_idx)
            gth_mask_image_list = []

        for (mask_name, cls) in zip(masks_name_list, range(1, class_num)):
            mask_image = slice_crop(np.load(mask_name), slice_idx)
            pred_mask_image_list.append(mask_generator(mask_image, colormap))
            if gth:
                gth_image = target_label.copy()
                gth_image[target_label != cls] = 0
                gth_mask_image_list.append(mask_generator(gth_image, colormap))

        pred_image = image_masks_merge(target_image, pred_mask_image_list)
        pred_name = '_'.join([target_name.strip('.npz'), 'pred', MODALITY]) + '.png'
        save_img(pred_image, pred_name)

        if gth:
            gth_image = image_masks_merge(target_image, gth_mask_image_list)
            gth_name = '_'.join([target_name.strip('.npz'), 'gth']) + '.png'
            save_img(gth_image, gth_name)


if __name__ == '__main__':
    main_process(gth=False, class_num=2)
    print('Done')