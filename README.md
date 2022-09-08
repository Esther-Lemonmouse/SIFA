## Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation

***
**Branch: Prostate** This job is the postgraduate project of _lemonmouse_. To find the original SIFA model, please view [cchen-cc: SIFA](https://github.com/cchen-cc/SIFA).
***

Tensorflow implementation of our unsupervised cross-modality domain adaptation framework. <br/>
This is the version of our [TMI paper](https://arxiv.org/abs/2002.02255). <br/>
Please refer to the branch [SIFA-v1](https://github.com/cchen-cc/SIFA/tree/SIFA-v1) for the version of our AAAI paper. <br/>

## Paper
[Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation](https://arxiv.org/abs/2002.02255)
<br/>
IEEE Transactions on Medical Imaging
<br/>
<br/>
<p align="center">
  <img src="figure/framework.png">
</p>

## Installation (modified)
* Install TensorFlow 1.10 and CUDA 9.0 (建议同时参考tensorflow官网)
* Clone this repo (原始版本是python 2的)
```
git clone https://github.com/Esther-Lemonmouse/SIFA/tree/prostate
cd SIFA
```

## Data Preparation
* Raw data needs to be written into `tfrecord` format to be decoded by `./data_loader.py`.
  
  prostate的数据集是我自己预处理过的，咱没有云盘，但是写了相关小工具去处理数据。链接在[这里(待补全)]()。
* Put `tfrecord` data of six domains into corresponding folders under `./data` accordingly. 和原文不同的地方是，由于我有多个数据集，因此在`./data`文件夹下，每个数据集都有自己的名字。
* Run `./create_datalist.py` to generate the datalists containing the path of each data.
* The `./data` directory working tree should be like the following format:
```
.
└── prostate
    ├── datalist
    │   └── data_prostate_testing_A.txt
    ├── testing
    │   └── A
    │       └── test_10.npz
    └── validation
        └── A
            └── slice_00000.tfrecords
```

## Train
* Modify the data statistics in data_loader.py according to the specifc dataset in use. Note that this is a very important step to correctly convert the data range to [-1, 1] for the network inputs and ensure the performance.

  这一步的具体实现可参看data_loader.py，我已经做过对应的case调整了。
* Modify paramter values in `./config_param.json`. 文件路径是个示例，参考着浅写。
* Run `./main.py` to start the training process.

## Evaluate
* (指原作者模型) Our trained models can be downloaded from [Dropbox](https://www.dropbox.com/sh/787kmmuhvh3e3yb/AAC4qxBJTWwQ1UMN5psrN96ja?dl=0).
  Note that the data statistics in evaluate.py need to be changed accordingly as specificed in the script.
* Specify the model path and test file path in `./evaluate.py`
* Run `./evaluate.py` to start the evaluation.

## Citation
If you find the code useful for your research, please cite our paper.
```
@article{chen2020unsupervised,
  title     = {Unsupervised Bidirectional Cross-Modality Adaptation via 
               Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation},
  author    = {Chen, Cheng and Dou, Qi and Chen, Hao and Qin, Jing and Heng, Pheng Ann},
  journal   = {arXiv preprint arXiv:2002.02255},
  year      = {2020}
}

@inproceedings{chen2019synergistic,
  author    = {Chen, Cheng and Dou, Qi and Chen, Hao and Qin, Jing and Heng, Pheng-Ann},
  title     = {Synergistic Image and Feature Adaptation: 
               Towards Cross-Modality Domain Adaptation for Medical Image Segmentation},
  booktitle = {Proceedings of The Thirty-Third Conference on Artificial Intelligence (AAAI)},
  pages     = {865--872},
  year      = {2019},
}
```

## Acknowledgement
Part of the code is revised from the [Tensorflow implementation of CycleGAN](https://github.com/leehomyc/cyclegan-1).

## Note
* The repository is being updated
* Contact: Cheng Chen (chencheng236@gmail.com)
