import numpy as np
import os
import tensorflow as tf


def npz2tfrecords(image_pth, tfrecord_pth):
    # image_pth = './example data/image_01.npz'
    # tfrecord_pth = './example data/image_01.tfrecords'

    npz_data = np.load(image_pth)
    data_vol_val = npz_data['arr_0']
    label_vol_val = npz_data['arr_1']
    dsize_dim0_val = data_vol_val.shape[0]
    dsize_dim1_val = data_vol_val.shape[1]
    dsize_dim2_val = data_vol_val.shape[2]
    lsize_dim0_val = label_vol_val.shape[0]
    lsize_dim1_val = label_vol_val.shape[1]
    lsize_dim2_val = label_vol_val.shape[2]

    writer = tf.python_io.TFRecordWriter(tfrecord_pth)

    feature = {'data_vol': tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(data_vol_val.tostring())])),
        'dsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim0_val])),
        'dsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim1_val])),
        'dsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim2_val])),
        'lsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim0_val])),
        'lsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim1_val])),
        'lsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim2_val])),
        'label_vol': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(label_vol_val.tostring())])), }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    writer.close()


if __name__=='__main__':
    source_dir = '/DataA/fmp22_shuxin_yao/src/dataset/Prostate/A/256-256-60/train'
    tf_dir = '../data/training_ct/A'
    filename_all_src = os.listdir(source_dir)
    for data in filename_all_src:
        tfname = data.split('/')[-1].split('.')[0] + '.tfrecords'
        tf_pth = os.path.join(tf_dir, tfname)
        data_pth = os.path.join(source_dir, data)
        npz2tfrecords(data_pth, tf_pth)
    print('Files in {0} has converted into {1}!\n'.format(source_dir, tf_dir))
    # npz2tfrecords()

