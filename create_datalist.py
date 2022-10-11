"""
本脚本用于生成数据目录文件便于训练
"""

import os
import numpy as np

#### brats2018 ####
SOURCE_DIR = './data/brats2018'
DESTINATION_DIR = './data/brats2018/datalist'

def make_datalist(data_fd, data_list):
    filename_all = os.listdir(data_fd)
    filename_all = [data_fd + '/' + filename + '\n' for filename in filename_all if
                    (filename.endswith('.tfrecords') or filename.endswith('.npz'))]

    np.random.shuffle(filename_all)
    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)


if __name__ == '__main__':
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR, exist_ok=True)

    data_tree = os.walk(SOURCE_DIR)

    for dirpath, dirname, filelist in data_tree:
        if not filelist:
            continue
        check_pth = os.path.join(dirpath, filelist[0])
        if os.path.isfile(check_pth):
            writelist = [dirpath+'/'+file+'\n' for file in filelist if file.endswith('tfrecords') or file.endswith('npz')]
        else:
            writelist = []

        if writelist:
            name_part = dirpath.split('/')[1:]
            txt_name = '_'.join(name_part) + '.txt'
            txt_dir = os.path.join(DESTINATION_DIR, txt_name)
            make_datalist(dirpath, txt_dir)

    print('Done')
