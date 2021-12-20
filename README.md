# DR-GAN: Automatic Radial Distortion Rectification Using Conditional GAN in Real-Time
## Introduction
This is official implementation for [DR-GAN](https://ieeexplore.ieee.org/document/8636975) (IEEE TCSVT'20).

[Kang Liao](https://kangliao929.github.io/), [Chunyu Lin](http://faculty.bjtu.edu.cn/8549/), [Yao Zhao](http://mepro.bjtu.edu.cn/zhaoyao/e_index.htm), [Moncef Gabbouj](https://www.tuni.fi/en/moncef-gabbouj)
> ### Problem
> Given a radial distortion image capture by wide-angle lens, DR-GAN aims to rectify the distortion and recover the realistic scene.
>  ### Features
>  * First generative framework for the distortion rectification
>  * One-stage rectification (Compared to previous two-stage rectification: distortion parameter estimation -> rectification)
>  * Label-free training (Directly learning the mapping between distorted structure and rectified structure)
>  * Real-time rectification (~66 FPS on NVIDIA GeForce RTX 2080Ti and ~26 FPS on TITAN X)

![](https://github.com/KangLiao929/DR-GAN/blob/main/img/1.png) 
## To be continued...

## Requirements
- Python 3.5.6 (or higher)
- Tensorflow 1.12.0
- Keras 2.2.4
- OpenCV
- numpy
- matplotlib
- scikit-image

## Installation

```bash
git clone https://github.com/KangLiao929/DR-GAN.git
cd DR-GAN/
```

## Getting Started & Testing

- Download the pretrained models through the following links ([generator](https://kangliao929.github.io/)), and unzip and put them into `weights/`. 
- To test images in a folder, you can call `test.py` with the opinion `--test_path` and `--load_models`. For example:

  ```bash
  python test.py --test_num 100 --test_path ./DR-GAN/dataset/test/ --load_models ./DR-GAN/weights/generator.h5 --write_path ./DR-GAN/dataset/pre/
  ```
  or write / modify `test.sh` according to your own needs, then execute this script as (Linux platform):  
  ```bash
  sh ./test.sh
  ```
The visual evaluations will be saved in the folder `./dataset/pre/`.

## Training
- Generate the training dataset
- To train DR-GAN, you can call `train.py` with the opinion `--train_path`. For example:
  ```shell
  python train.py --train_path ./DR-GAN/dataset/train/ --batch_size 16 --gpu "0"
  ```
  or write / modify `train.sh` according to your own needs, then execute this script as:  
  ```bash
  sh ./train.sh
  ```

## Citation

If our method is useful for your research, please consider citing:

    @ARTICLE{liao2020drgan,
      author={Liao, Kang and Lin, Chunyu and Zhao, Yao and Gabbouj, Moncef},
      journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
      title={DR-GAN: Automatic Radial Distortion Rectification Using Conditional GAN in Real-Time}, 
      year={2020},
      volume={30},
      number={3},
      pages={725-733}}
