#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File       vis.py
@Author     Shuxin Yao
@Email      yaosx_job@outlook.com
@CreateTime 2022/07/22
@Function   输入一组numpy格式的分割结果 (来自evaluation和ground truth)，获取可视化结果。

获取分割结果的方法 (需使用evaluate.py)：
1. 使用debug模式在以下代码处添加断点：
  约line234处: 
    for c in range(1, self._num_cls):
                    pred_test_data_tr = tmp_pred.copy()
                    pred_test_data_tr[pred_test_data_tr != c] = 0   # 模型预测结果，仅含有当前类别 (即c类) 的分割标签，其他部分修改为0

                    pred_gt_data_tr = label.copy()
                    pred_gt_data_tr[pred_gt_data_tr != c] = 0   # ground truth仅含对应类别的分割结果

2. 使用np.savez()方法，将分割结果保存为'[DOMAIN]_[CASE_NUMBER].npz'。sample: np.savez('A_test_3.npz', data, gt_mask, pred_mask)

3. 修改MODALITY的值，注意要让名称匹配[source2target]的模式
"""
import os
import cv2
import numpy as np
import argparse

MODALITY = 'A2B-mixup-alpha0.2'

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


def main_process(keywords, gth=False, class_num=2):
    os.chdir('../visualization/')
    rng = np.random.default_rng(17)
    colormap = colormap_generator(class_num, rng)
    target_img_list = [fn for fn in os.listdir() if fn.endswith('npz')]
    if not target_img_list:
        raise ValueError('Please move the test images manually into directory "visualization". ')

    for target_name in target_img_list:
        target = np.load(target_name)
        target_image = target['arr_0']

        slice_idx = np.floor(target_image.shape[2] / 2).astype(np.int16)
        target_image_norm = slice_crop(target_image, slice_idx)
        target_image = ((target_image_norm+target_image_norm.min())
                        *(255/(target_image_norm.max()-target_image_norm.min()))).astype(np.uint8)
        pred_mask_image_list = []
        if gth:
            target_label = target['arr_1']
            target_label = slice_crop(target_label, slice_idx)
            gth_mask_image_list = []

        for (mask_name, cls) in zip(target_img_list, range(1, class_num)):
            mask_image = slice_crop(target['arr_2'], slice_idx)
            pred_mask_image_list.append(mask_generator(mask_image, colormap))
            if gth:
                gth_mask_image = target_label.copy()
                gth_mask_image[gth_mask_image != cls] = 0
                gth_mask_image_list.append(mask_generator(gth_mask_image, colormap))

        pred_image = image_masks_merge(target_image, pred_mask_image_list)
        pred_name = '_'.join([target_name.strip('.npz'), 'pred', keywords]) + '.png'
        save_img(pred_image, pred_name)

        if gth:
            gth_image = image_masks_merge(target_image, gth_mask_image_list)
            gth_name = '_'.join([target_name.strip('.npz'), 'gth']) + '.png'
            save_img(gth_image, gth_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='请设置类别，关键字和是否打印gt')
    parser.add_argument('-g', '--gth', type=bool, default=False, help='是否打印gt, default is False')
    parser.add_argument('-c', '--classes', type=int, default=2, required=True, help='必须设置类别')
    parser.add_argument('-k', '--keywords', default=MODALITY, help='生成图片的关键字')
    args = parser.parse_args()
    gth = args.gth
    classes = args.classes
    keywords = args.keywords

    main_process(keywords, gth, classes)
    print('Done')