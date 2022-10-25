#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@File       mixup.py
@Author     Shuxin Yao
@Email      yaosx_job@outlook.com
@CreateTime 2022/10/24
@Function   简单MIXUP增广方法的实现
            基本都是照着底下的keras example写的，我太菜了
---------------------------------------
@References - Original Paper
                @misc{https://doi.org/10.48550/arxiv.1710.09412,
                    doi = {10.48550/ARXIV.1710.09412},
                    url = {https://arxiv.org/abs/1710.09412},
                    author = {Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N. and Lopez-Paz, David},
                    keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
                    title = {mixup: Beyond Empirical Risk Minimization},
                    publisher = {arXiv},
                    year = {2017},
                    copyright = {arXiv.org perpetual, non-exclusive license}
                }
            - Implementation Site: https://keras.io/examples/vision/mixup/
"""
# import tensorflow as tf
import numpy as np

# def sample_beta_distribution(size, concentration_1=0.2, concentration_2=0.2):
#     """
#     since tensorflow has no direct implementation of beta distribution, for TF implementation, we have to use gamma
#     function firstly.
#     The relationship between Beta and Gamma distribution: https://blog.csdn.net/xhf0374/article/details/53946146
#     """
#     sample_gamma_1 = tf.random_gamma([size], concentration_1)
#     sample_gamma_2 = tf.random_gamma([size], concentration_2)
#     return sample_gamma_1 / (sample_gamma_1 + sample_gamma_2)


def mix_up_sifa(img_batch_1, img_batch_2, label_batch_1, label_batch_2, alpha=0.2):
    batch_size = img_batch_1.shape[0]
    total_repeat = img_batch_1.shape[1]*img_batch_1.shape[2]*img_batch_1.shape[3]

    lbd = np.random.beta(alpha, alpha, batch_size).astype(np.float32)
    lbd = lbd.repeat(total_repeat).reshape(img_batch_1.shape)

    mix_img_batch = img_batch_1*lbd + img_batch_2*(1-lbd)
    mix_label_batch = label_batch_1*lbd + label_batch_2*(1-lbd)

    return mix_img_batch, mix_label_batch
