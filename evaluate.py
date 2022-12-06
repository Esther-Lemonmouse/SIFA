"""Code for testing SIFA."""
import json
import numpy as np
import os
import medpy.metric.binary as mmb

import tensorflow as tf

import model
from stats_func import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # in local file, this is not useful

CHECKPOINT_PATH = './output/yyyymmdd-hhmmss/xxx' # model path
BASE_FID = '' # folder path of test files
TESTFILE_FID = './data/prostate/datalist/xxx.txt' # path of the .txt file storing the test filenames
########################## Prostate modality options ##########################
# A B C D E F
###############################################################################
TEST_MODALITY = ' ' # the modality name you want to be tested
KEEP_RATE = 1.0
IS_TRAINING = False
BATCH_SIZE = 30 # Batch size is based on how many batches (i.e. 3xBATCH_SIZE) in your each test case

data_size = [256, 256, 1]
label_size = [256, 256, 1]

#### prostate 二分类的图片 ####
contour_map = {
    'bg': 0,
    'prostate': 1,
    }

#### original cardiac ####
# contour_map = {
#     "bg": 0,
#     "la_myo": 1,
#     "la_blood": 2,
#     "lv_blood": 3,
#     "aa": 4,
# }


class SIFA:
    """The SIFA module."""

    def __init__(self, config):

        self.keep_rate = KEEP_RATE
        self.is_training = IS_TRAINING
        self.checkpoint_pth = CHECKPOINT_PATH
        self.batch_size = BATCH_SIZE

        self._pool_size = int(config['pool_size'])
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])

        self.base_fd = BASE_FID
        self.test_fid = TESTFILE_FID

    def model_setup(self):

        self.input_a = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="input_B")
        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_B")
        self.gt_a = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                self._num_cls
            ], name="gt_A")
        self.gt_b = tf.placeholder(
            tf.float32, [
                self.batch_size,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                self._num_cls
            ], name="gt_B")

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(inputs, skip=self._skip, is_training=self.is_training, keep_rate=self.keep_rate)

        self.pred_mask_b = outputs['pred_mask_b']

        self.predicter_b = tf.nn.softmax(self.pred_mask_b)
        self.compact_pred_b = tf.argmax(self.predicter_b, 3)
        self.compact_y_b = tf.argmax(self.gt_b, 3)

    def read_lists(self, fid):
        """read test file list """

        with open(fid, 'r') as fd:
            _list = fd.readlines()

        my_list = []
        for _item in _list:
            # my_list.append(self.base_fd + '/' + _item.split('\n')[0]) # 绝对目录
            my_list.append(_item.split('\n')[0])    # 相对目录
        return my_list

    def label_decomp(self, label_batch):
        """decompose label for one-hot encoding """

        _batch_shape = list(label_batch.shape)
        _vol = np.zeros(_batch_shape)
        _vol[label_batch == 0] = 1
        _vol = _vol[..., np.newaxis]
        for i in range(self._num_cls):
            if i == 0:
                continue
            _n_slice = np.zeros(label_batch.shape)
            _n_slice[label_batch == i] = 1
            _vol = np.concatenate( (_vol, _n_slice[..., np.newaxis]), axis = 3 )
        return np.float32(_vol)

    def test(self):
        """Test Function."""

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        test_list = self.read_lists(self.test_fid)

        # eval时，为了存图，限制一下GPU使用量
        ############原始代码#############
        # with tf.Session() as sess:
        ################################
        gpu_config = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)) as sess:
        ################################
            sess.run(init)

            saver.restore(sess, self.checkpoint_pth)

            dice_list = []
            assd_list = []
            for idx_file, fid in enumerate(test_list):
                _npz_dict = np.load(fid)
                data = _npz_dict['arr_0']
                label = _npz_dict['arr_1']

                # This is to make the orientation of test data match with the training data
                # Set to False if the orientation of test data has already been aligned with the training data
                if False:  # 我的数据处理过程中训练集和测试集的方向是一致的，无需进行翻转
                    data = np.flip(data, axis=0)
                    data = np.flip(data, axis=1)
                    label = np.flip(label, axis=0)
                    label = np.flip(label, axis=1)

                tmp_pred = np.zeros(label.shape)

                frame_list = [kk for kk in range(data.shape[2])]
                for ii in range(int(np.floor(data.shape[2] // self.batch_size))):
                    data_batch = np.zeros([self.batch_size, data_size[0], data_size[1], data_size[2]])
                    label_batch = np.zeros([self.batch_size, label_size[0], label_size[1]])
                    for idx, jj in enumerate(frame_list[ii * self.batch_size: (ii + 1) * self.batch_size]):
                        data_batch[idx, ...] = np.expand_dims(data[..., jj].copy(), 2)  ###3改成2
                        label_batch[idx, ...] = label[..., jj].copy()
                    label_batch = self.label_decomp(label_batch)

                    ############################################################
                    # 原始SIFA文件设置: using Cardiac set
                    # if TEST_MODALITY=='CT':
                    #     if USE_newstat:
                    #         data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -2.8), np.subtract(3.2, -2.8)), 2.0),1) # {-2.8, 3.2} need to be changed according to the data statistics
                    #     else:
                    #         data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -1.9), np.subtract(3.0, -1.9)), 2.0),1) # {-1.9, 3.0} need to be changed according to the data statistics
                    # elif TEST_MODALITY=='MR':
                    #     data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -1.8), np.subtract(4.4, -1.8)), 2.0),1)  # {-1.8, 4.4} need to be changed according to the data statistics
                    ############################################################
                    ############################################################
                    # prostate 数据集相关最值参数设置
                    # A: {-3.1, 4.2}
                    # B: {-3.0, 3.8}
                    # C: {-3.2, 4.3}
                    # D: {-3.0, 4.1}
                    # E: {-3.8, 6.3}
                    # F: {-3.4, 4.9}
                    ############################################################
                    if TEST_MODALITY == 'A':
                        data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -3.1), np.subtract(4.2, -3.1)), 2.0), 1)
                    elif TEST_MODALITY == 'B':
                        data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -3.0), np.subtract(3.8, -3.0)), 2.0), 1)
                    elif TEST_MODALITY == 'C':
                        data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -3.2), np.subtract(4.3, -3.2)), 2.0), 1)
                    elif TEST_MODALITY == 'D':
                        data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -3.0), np.subtract(4.1, -3.0)), 2.0), 1)
                    elif TEST_MODALITY == 'E':
                        data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -3.8), np.subtract(6.3, -3.8)), 2.0), 1)
                    elif TEST_MODALITY == 'F':
                        data_batch = np.subtract(np.multiply(np.divide(np.subtract(data_batch, -3.4), np.subtract(4.9, -3.4)), 2.0), 1)
                    else:
                        raise NameError('Unexpected test modality. It should an alphabet in A-F.')
                    ############################################################

                    compact_pred_b_val = sess.run(self.compact_pred_b, feed_dict={self.input_b: data_batch, self.gt_b: label_batch})

                    for idx, jj in enumerate(frame_list[ii * self.batch_size: (ii + 1) * self.batch_size]):
                        tmp_pred[..., jj] = compact_pred_b_val[idx, ...].copy()

                for c in range(1, self._num_cls):
                    pred_test_data_tr = tmp_pred.copy()
                    pred_test_data_tr[pred_test_data_tr != c] = 0

                    pred_gt_data_tr = label.copy()
                    pred_gt_data_tr[pred_gt_data_tr != c] = 0

                    dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                    assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))

            # dice_arr = 100 * np.reshape(dice_list, [4, -1]).transpose()
            dice_arr = 100 * np.reshape(dice_list, [1, -1]).transpose() # prostate 共两类

            dice_mean = np.mean(dice_arr, axis=1)
            dice_std = np.std(dice_arr, axis=1)

            print('Dice:')
            print('Prostate:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
            # print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
            # print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
            # print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
            # print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
            print('Mean:%.1f' % np.mean(dice_mean))

            # assd_arr = np.reshape(assd_list, [4, -1]).transpose()
            assd_arr = np.reshape(assd_list, [1, -1]).transpose()   # prostate 共两类

            assd_mean = np.mean(assd_arr, axis=1)
            assd_std = np.std(assd_arr, axis=1)

            print('ASSD:')
            # print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
            # print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
            # print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
            # print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
            print('Mean:%.1f' % np.mean(assd_mean))


def main(config_filename):

    with open(config_filename) as config_file:
        config = json.load(config_file)

    sifa_model = SIFA(config)
    sifa_model.test()


if __name__ == '__main__':
    print('checkpoint: {0:}\ntestfile name: {1:}'.format(CHECKPOINT_PATH, TESTFILE_FID))
    main(config_filename='./config_param.json')
