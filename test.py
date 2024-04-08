import numpy as np
import time
import argparse
from glob import glob
import cv2
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from network.model import generator
from network.utils import *
import os

parser = argparse.ArgumentParser()
parser.add_argument("--test_num", type=int, default=1000, help="test data num")
parser.add_argument("--test_path", type=str, default="/test/", help="path of the test dataset")
parser.add_argument("--write_path", type=str, default="/results/test/", help="path of saving predicted images")
parser.add_argument("--load_models", default='generator.h5', help="where to load models")
parser.add_argument("--gpu", type=str, default="8", help="gpu number")
parser.add_argument("--img_w", type=int, default=256, help="width of images")
parser.add_argument("--img_h", type=int, default=256, help="height of images")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.write_path, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

def rectify_real_image():
    # load weights
    g = generator()
    g.load_weights(opt.load_models, by_name = True)

    # path
    path1 = opt.test_path + "/*.*" # distorted image
    loc_list1 = glob(path1)

    for i in range(opt.test_num):
        
        src = cv2.imread(loc_list1[i])
        src = cv2.resize(src, (opt.img_w, opt.img_h))
        
        x_test = src.astype('float32')
        x_test = (x_test - 127.5) / 127.5

        x_test = np.expand_dims(x_test, axis = 0)

        s1 = time.time()
        rec = g.predict(x = x_test)
        rec = np2img(rec)
        s2 = time.time()
        print("test time: ", s2 - s1)

        _, filename = os.path.split(loc_list1[i])
        file = opt.write_path + filename
        print(file)
        cv2.imwrite(file, rec)

def rectify_image():
    # load weights
    g = generator()
    g.load_weights(opt.load_models, by_name = True)

    # path
    path1 = opt.test_path + "A/*.*" # distorted image
    path2 = opt.test_path + "B/*.*" # ground truth
    loc_list1 = glob(path1)
    loc_list2 = glob(path2) 

    avg_ssim = 0
    avg_psnr = 0
    for i in range(opt.test_num):
        
        src = cv2.imread(loc_list1[i])
        gt = cv2.imread(loc_list2[i])
        src = cv2.resize(src, (opt.img_w, opt.img_h))
        gt = cv2.resize(gt, (opt.img_w, opt.img_h))
        
        x_test = src.astype('float32')
        x_test = (x_test - 127.5) / 127.5

        x_test = np.expand_dims(x_test, axis = 0)

        s1 = time.time()
        rec = g.predict(x = x_test)
        s2 = time.time()
        print("test time: ", s2 - s1)

        rec = np2img(rec)
        s = ssim(gt, rec, multichannel = True)
        p = psnr(gt, rec)
        avg_ssim += s
        avg_psnr += p

        _, filename = os.path.split(loc_list1[i])
        file = opt.write_path + filename
        print(file)
        cv2.imwrite(file, rec)
        
    print("ssim: ", avg_ssim / opt.test_num)
    print("psnr: ", avg_psnr / opt.test_num) 
    
if __name__ == "__main__":
    rectify_real_image()
