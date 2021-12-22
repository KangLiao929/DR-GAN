# DR-GAN: Automatic Radial Distortion Rectification Using Conditional GAN in Real-Time
## Introduction
This is the official implementation for [DR-GAN](https://ieeexplore.ieee.org/document/8636975) (IEEE TCSVT'20).

[Kang Liao](https://kangliao929.github.io/), [Chunyu Lin](http://faculty.bjtu.edu.cn/8549/), [Yao Zhao](http://mepro.bjtu.edu.cn/zhaoyao/e_index.htm), [Moncef Gabbouj](https://www.tuni.fi/en/moncef-gabbouj)
> ### Problem
> Given a radial distortion image capture by wide-angle lens, DR-GAN aims to rectify the distortion and recover the realistic scene.
>  ### Features
>  * First generative framework for the distortion rectification
>  * One-stage rectification (Compared to previous two-stage rectification: distortion parameter estimation -> rectification)
>  * Label-free training (Directly learning the mapping between distorted structure and rectified structure)
>  * Real-time rectification (~66 FPS on NVIDIA GeForce RTX 2080Ti and ~26 FPS on TITAN X)

![](https://github.com/KangLiao929/DR-GAN/blob/main/img/1.png) 

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

## Dataset

We synthesize the radial distortion image dataset including the training and test data, based on the polynomial camera model (four distortion parameters: k1, k2, k3, and k4 involved). In training/test dataset, the radial distortion image and its corresponding GT are provided in 'A' folder and 'B' folder, respectively. 

The whole synthesized dataset is available [here](https://drive.google.com/drive/folders/1VAKPe3nmFVRIxVv35ueTkB2JXCsMvdfS?usp=sharing), which can serve as a benchmark in the field of distortion rectification. To our knowledge, there is no released distortion image dataset for both training and performance evaluation.

## Getting Started & Testing

- Download the pretrained models through the following links ([generator](https://drive.google.com/file/d/1dmEpGbsHIdmMDtdPxcFPPhRl9TL0TquJ/view?usp=sharing)), and put them into `weights/`. 
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
- Generate the training dataset or download our synthesized dataset into the path `dataset/train/`.
- To train DR-GAN, you can call `train.py` with the opinion `--train_path`. For example:
  ```shell
  python train.py --train_path ./DR-GAN/dataset/train/ --batch_size 16 --gpu "0"
  ```
  or write / modify `train.sh` according to your own needs, then execute this script as:  
  ```bash
  sh ./train.sh
  ```

## Limitations

Compared to previous parameter-based methods, our DR-GAN is the first attempt at the generation-based solution and achieves real-time rectification. However, it has the following limitations which could be the possible effort directions for future works.

- **Blurred rectified details**. Due to the vanilla skip-connection and many downsampling layers in the generator network, our rectified images suffer from visually unpleasing details. [Our solution](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Progressively_Complementary_Network_for_Fisheye_Image_Rectification_Using_Appearance_Flow_CVPR_2021_paper.pdf)
- **Lack of camera parameters**. Due to directly learning the geometric transformation mapping, DR-GAN does not rely on parameter estimation. However, for other research fields such as camera calibration and SfM, the camera parameters are crucial. [Our solution](https://ieeexplore.ieee.org/document/9366359)


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
