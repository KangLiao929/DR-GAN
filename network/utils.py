import os
import numpy as np
from glob import glob
import cv2
from matplotlib import pyplot as plt

img_w = 256
img_h = 256
img_channels = 3

def plot_images(images, save2file, path, name, step):
    filename = path + "%d_" % step + name + ".png"

    plt.figure(figsize = (10, 10))
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        image = images[i, :, :]
        image = np2img(image)
        plt.imshow(image)
        plt.axis('off')
    plt.tight_layout()
    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()
    
def save_all_weights(g, d, save_dir, epoch_number):
    g.save_weights(os.path.join(save_dir, 'generator_{}.h5'.format(epoch_number)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)

def get_data(img_num, path):
    path1 = path + "A/*.*" # distorted image
    path2 = path + "B/*.*"  # gt rectified image      
    loc_list1 = glob(path1)
    loc_list2 = glob(path2) 
    
    src = np.zeros((img_num, img_w, img_h, img_channels)) 
    gt = np.zeros((img_num, img_w, img_h, img_channels)) 
    
    for i in range(img_num):
        # image reading
        img1 = cv2.imread(loc_list1[i])
        img2 = cv2.imread(loc_list2[i])
                
        img1 = np.reshape(img1, (img_w, img_h, img_channels))
        img2 = np.reshape(img2, (img_w, img_w, img_channels))
        
        src[i, :, :] = img1
        gt[i, :, :] = img2
    
    src = src.astype('float32')
    src = (src - 127.5) / 127.5
    gt = gt.astype('float32')
    gt = (gt - 127.5) / 127.5
    return src, gt

def load_batch(train_num, batch_size, path):   
    path1 = path + "A/*.*" # distorted image
    path2 = path + "B/*.*"  # gt rectified image          
    loc_list1 = glob(path1)
    loc_list2 = glob(path2) 
    
    src = np.zeros((batch_size, img_w, img_h, img_channels)) 
    gt = np.zeros((batch_size, img_w, img_h, img_channels))   
    
    n_batches = int(train_num / batch_size)
    
    for i in range(n_batches):
        for j in range(batch_size):
            # image reading
            img1 = cv2.imread(loc_list1[i * batch_size + j])
            img2 = cv2.imread(loc_list2[i * batch_size + j])    
            
            src[j, :, :] = img1
            gt[j, :, :] = img2
        
        src = src.astype('float32')
        src = (src - 127.5) / 127.5
        gt = gt.astype('float32')
        gt = (gt - 127.5) / 127.5
        yield src, gt


def np2img(img):
    img = img * 127.5 + 127.5
    img = img.astype('uint8')
    img = np.reshape(img, [img_w, img_h, img_channels])
    return img