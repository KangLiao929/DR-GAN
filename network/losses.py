import keras.backend as K
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras import objectives
import tensorflow as tf

image_shape = (256, 256, 3)
img_width = 256
img_height = 256


def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis = -1)
    
def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis = -1)

def psnr_loss(y_true, y_pred):
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)
    
def perceptual_loss(y_true, y_pred):
    low = perceptual_low_loss(y_true, y_pred)
    high = perceptual_high_loss(y_true, y_pred)
    return 0.2 * low + 0.8 * high
    
def perceptual_low_loss(y_true, y_pred):
    vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = image_shape)
    loss_model = Model(inputs = vgg.input, outputs = vgg.get_layer('block1_conv2').output)
    loss_model.trainable = False
    return K.sum(K.square(loss_model(y_true) - loss_model(y_pred))) / (img_width * img_height)
    
def perceptual_high_loss(y_true, y_pred):
    vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = image_shape)
    loss_model = Model(inputs = vgg.input, outputs = vgg.get_layer('block5_conv2').output)
    loss_model.trainable = False
    return K.sum(K.square(loss_model(y_true) - loss_model(y_pred))) / (img_width * img_height)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
    
    