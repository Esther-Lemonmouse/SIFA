import os
import numpy as np

test_dir = '/DataA/fmp22_shuxin_yao/src/dataset/SIFA-Preprocess'  # remote
# test_dir = './data'     # local


def make_datalist(data_fd, data_list):
    filename_all = os.listdir(data_fd)
    filename_all = [data_fd + '/' + filename + '\n' for filename in filename_all if
                    (filename.endswith('.tfrecords') or filename.endswith('.npz'))]

    np.random.shuffle(filename_all)
    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)


if __name__ == '__main__':
    # data_fd_prefix = './data/training_ct'
    # data_fd_prefix = './data/validation_ct'
    # data_fd_prefix = './data/training_mr'
    data_fd_prefix = './data/validation_mr'
    data_list_dir = './data/datalist'

    fid = os.path.join(data_fd_prefix, 'B')
    # list_file = data_list_dir + '/training_ct_A.txt'
    # list_file = data_list_dir + '/validation_ct_A.txt'
    # list_file = data_list_dir + '/training_mr_B.txt'
    list_file = data_list_dir + '/validation_mr_B.txt'

    # make_datalist('/DataA/fmp22_shuxin_yao/src/dataset/Prostate/B/256-256-60/train', './data/datalist/train_prostate_B.txt')
    make_datalist('/DataA/fmp22_shuxin_yao/src/dataset/Prostate/temptest/A', './data/datalist/test_ct_A.txt')
    # make_datalist(fid, list_file)
