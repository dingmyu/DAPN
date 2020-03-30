## Introduction

The framework is implemented and tested with Ubuntu 16.04, CUDA 8.0/9.0, Python 3, Pytorch 0.4/1.0/1.1, NVIDIA TITANX GPU. 

## Requirements

- **Cuda & Cudnn & Python & Pytorch**

    This project is tested with CUDA 8.0/9.0, Python 3, Pytorch 0.4/1.0, NVIDIA TITANX GPUs.

    Please install proper CUDA and CUDNN version, and then install Anaconda3 and Pytorch. Almost all the packages we use are covered by Anaconda.

- **My settings**

  ```shell
  source ~/anaconda3/bin/activate (python 3.6.5)
	(base)  pip list
	torch                              0.4.1
	torchvision                        0.2.2.post3
	numpy                              1.18.1
	numpydoc                           0.8.0
	numba                              0.42.0
	opencv-python                      4.0.0.21
  ```


## Data preparation

Download and unzip the datasets: **MiniImageNet, TieredImageNet, DomainNet**.

Here we provide the datasets of target domain in Google Drive, [miniImageNet](https://drive.google.com/file/d/1Ai0070r-eZoJb_4vamYipYKbX24jkjFg), [tieredImageNet](https://drive.google.com/file/d/18xpXtAAm_onIcwxtesRaXiHt99hm7ln6).

Format:
(E.g. mini-imagenet)
  ```shell
MINI_DIR/
    --  train/
        --  n01532829/
        --  n01558993/
        ...
    --  train_new_domain/
    --  val/
    --  val_new_domain/
    --  test/
    --  test_new_domain/
  ```


## Training

First set the dataset path `MINI_DIR/, TIERED_DIR/, DOMAIN_DIR/` for the three datasets.

For each dataset, we use its training set to train a pre-trained model (e.g. tiered-imagenet).

``` 
cd pretrain/
python -u main_resnet.py --epochs 50 --batch_size 1024  --dir_path TIERED_DIR 2>&1 | tee log.txt &
```

We then use the corresponding pre-trained model to train on each dataset. (e.g. mini-imagenet)

```
python -u train_cross.py --gpu_id 0 --net ResNet50 --dset mini-imagenet --s_dset_path MINI_DIR --fsl_test_path MINI_DIR --shot 5 --train-way 16 --pretrained 'mini_checkpoint.pth.tar' --output_dir mini_way_16
```


## Testing

```
python -u test.py --load MODEL_PATH --root MINI_DIR
```

