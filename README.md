# DR-GAN: Automatic Radial Distortion Rectification Using Conditional GAN in Real-Time
## Introduction
This is official implementation for [DR-GAN](https://ieeexplore.ieee.org/document/8636975) (IEEE TCSVT'20).

[Kang Liao](https://kangliao929.github.io/), [Chunyu Lin](http://faculty.bjtu.edu.cn/8549/), [Yao Zhao](http://mepro.bjtu.edu.cn/zhaoyao/e_index.htm), [Moncef Gabbouj](https://www.tuni.fi/en/moncef-gabbouj)
> ### Problem
> Given a radial distortion image capture by wide-angle lens, DR-GAN aims to rectify the distortion and recover the realistic scene.
>  ### Features
>  * The first generative framework for the distortion rectification
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
