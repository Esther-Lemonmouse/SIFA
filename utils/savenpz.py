# 这个脚本现在不需要了
import os.path
import numpy as np
import medpy.io as medio


def nii2npz():
    data_nii_pth = '/home/lemonmouse/Data/share/dataset/prostate/utils/resampled nifti/A/256-256-60/validation_image'
    label_nii_pth = '/home/lemonmouse/Data/share/dataset/prostate/utils/resampled nifti/A/256-256-60/validation_label'
    npz_pth = '../data/test_ct/A'

    data_dir = [os.path.join(data_nii_pth, name) for name in os.listdir(data_nii_pth)]
    label_dir = [os.path.join(label_nii_pth, name) for name in os.listdir(label_nii_pth)]
    data_dir.sort()
    label_dir.sort()

    for (d, l) in zip(data_dir, label_dir):
        data_arr, _ = medio.load(d)
        label_arr, _ = medio.load(l)
        npz_dir = npz_pth + '/' + d.split('/')[-1].split('.')[0] + '.npz'
        np.savez(npz_dir, data_arr, label_arr)

    # data_arr, _ = medio.load(data_nii_pth)
    # label_arr, _ = medio.load(label_nii_pth)

    # np.savez(npz_pth, data_arr, label_arr)


if __name__=="__main__":
    nii2npz()