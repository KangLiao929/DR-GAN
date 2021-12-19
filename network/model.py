from keras.layers import Input, Activation, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras

# hyper-parameter
weight_decay = 0.01
ndf = 64
img_shape = (256, 256, 3)

def generator():
    # 256x256x3
    inputs = Input(shape = img_shape)
    # Convolution
    # 128x128x64
    conv1 = Conv2D(64, (4, 4), strides = (2, 2), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    # 64x64x128
    conv2 = Conv2D(128, (4, 4), strides = (2, 2), padding = 'same',
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(0.2)(conv2)
    # 32x32x256
    conv3 = Conv2D(256, (4, 4), strides = (2, 2), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(0.2)(conv3)
    # 16x16x512
    conv4 = Conv2D(512, (4, 4), strides = (2, 2), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(0.2)(conv4)
    # 8x8x512
    conv5 = Conv2D(512, (4, 4), strides = (2, 2),padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(0.2)(conv5)
    # 4x4x512
    conv6 = Conv2D(512, (4, 4), strides = (2, 2),padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(0.2)(conv6)
    # 2x2x512
    conv7 = Conv2D(512, (4, 4), strides = (2, 2),padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(conv6)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(0.2)(conv7)
    # 1x1x512
    conv8 = Conv2D(512, (4, 4), strides = (2, 2), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(0.2)(conv8)
    
    # Deconvolution
    # 2x2x512
    up1 = UpSampling2D()(conv8)
    de_conv1 = Conv2D(512, (4, 4), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(up1)
    de_conv1 = BatchNormalization()(de_conv1)
    de_conv1 = Dropout(0.5)(de_conv1)
    de_conv1 = LeakyReLU(0.2)(de_conv1)
    merge1 = concatenate([de_conv1, conv7], axis = 3)
    # 4x4x512
    up2 = UpSampling2D()(merge1)
    de_conv2 = Conv2D(512, (4, 4), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(up2)
    de_conv2 = BatchNormalization()(de_conv2)
    de_conv2 = Dropout(0.5)(de_conv2)
    de_conv2 = LeakyReLU(0.2)(de_conv2)
    
    merge2 = concatenate([de_conv2, conv6], axis = 3)      
    # 8x8x512
    up3 = UpSampling2D()(merge2)
    de_conv3 = Conv2D(512, (4, 4), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(up3)
    de_conv3 = BatchNormalization()(de_conv3)
    de_conv3 = Dropout(0.5)(de_conv3)
    de_conv3 = LeakyReLU(0.2)(de_conv3)
    
    merge3 = concatenate([de_conv3, conv5], axis = 3)

    # 16x16x512
    up4 = UpSampling2D()(merge3)
    de_conv4 = Conv2D(512, (4, 4), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(up4)
    de_conv4 = BatchNormalization()(de_conv4)
    de_conv4 = LeakyReLU(0.2)(de_conv4)
    merge4 = concatenate([de_conv4, conv4], axis = 3)
    # 32x32x256
    up5 = UpSampling2D()(merge4)
    de_conv5 = Conv2D(256, (4, 4), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(up5)
    de_conv5 = BatchNormalization()(de_conv5)
    de_conv5 = LeakyReLU(0.2)(de_conv5)
    merge5 = concatenate([de_conv5, conv3], axis = 3)
    # 64x64x128
    up6 = UpSampling2D()(merge5)
    de_conv6 = Conv2D(128, (4, 4), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(up6)
    de_conv6 = BatchNormalization()(de_conv6)
    de_conv6 = LeakyReLU(0.2)(de_conv6)
    merge6 = concatenate([de_conv6, conv2], axis = 3)
    # 128x128x64
    up7 = UpSampling2D()(merge6)
    de_conv7 = Conv2D(64, (4, 4), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(up7)
    de_conv7 = BatchNormalization()(de_conv7)
    de_conv7 = LeakyReLU(0.2)(de_conv7)
    merge7 = concatenate([de_conv7, conv1], axis = 3)             
    # 256x256x3
    up8 = UpSampling2D()(merge7)
    de_conv8 = Conv2D(3, (4, 4), padding = 'same', 
                    kernel_regularizer = keras.regularizers.l2(weight_decay),
                    kernel_initializer = 'he_normal')(up8)
    de_conv8 = BatchNormalization()(de_conv8)
                    
    #outputs    
    outputs = Activation('tanh')(de_conv8)
    
    model = Model(input = inputs, output = outputs)
    return model
    
  
def discriminator():
    n_layers, use_sigmoid = 3, False
    inputs = Input(shape = img_shape)
    
    x = Conv2D(filters = ndf, kernel_size = (4, 4), strides = 2, 
                kernel_regularizer = keras.regularizers.l2(weight_decay),
                kernel_initializer = 'he_normal', padding = 'same')(inputs)
    x = LeakyReLU(0.2)(x)
    
    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
        x = Conv2D(filters = ndf * nf_mult, kernel_size = (4, 4), strides = 2,
                kernel_regularizer = keras.regularizers.l2(weight_decay),
                kernel_initializer = 'he_normal', padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
    
    nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
    x = Conv2D(filters = ndf * nf_mult, kernel_size = (4, 4), strides = 1, 
                kernel_regularizer = keras.regularizers.l2(weight_decay),
                kernel_initializer = 'he_normal', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(filters = 1, kernel_size = (4, 4), strides = 1, 
                kernel_regularizer = keras.regularizers.l2(weight_decay),
                kernel_initializer = 'he_normal', padding = 'same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)
    
    x = Flatten()(x)
    x = Dense(1024, activation = 'relu')(x)
    x = Dense(1, activation = 'sigmoid')(x)
    
    model = Model(inputs = inputs, outputs = x)
    return model
    
   
def DRGAN(g, d):
    input = Input(shape = img_shape)
    
    pre = g(input)
    d_pre = d(pre)
    
    model = Model(inputs = input, outputs = [pre, d_pre])
    return model   

  
        
    
    