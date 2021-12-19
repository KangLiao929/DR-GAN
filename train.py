import os
import argparse
import numpy as np

from network.utils import *
from network.losses import wasserstein_loss, perceptual_loss
from network.model import generator, discriminator, DRGAN
from keras.optimizers import Adam
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim


parser = argparse.ArgumentParser()
parser.add_argument("--epoch_num", type=int, default=200, help="epoch to start training from")
parser.add_argument("--train_num", type=int, default=30000, help="training data num")
parser.add_argument("--test_num", type=int, default=500, help="test data num")
parser.add_argument("--train_path", type=str, default="/data/cylin/lk/OP/dataset_mask/train/", help="path of the train dataset")
parser.add_argument("--test_path", type=str, default="/data/cylin/lk/OP/dataset_mask/test/", help="path of the test dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=1e-4, help="adam: learning rate of generator")
parser.add_argument("--lr_d", type=float, default=1e-4, help="adam: learning rate of discriminator")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--ep", type=float, default=1e-08, help="adam: epsilon")
parser.add_argument("--img_size", type=int, default=256, help="training image size 256 or 512")
parser.add_argument("--save_interval", type=int, default=5, help="interval between saving image samples and checkpoints")
parser.add_argument("--lambda_gen", type=float, default=20, help="generator loss weight")
parser.add_argument("--lambda_dis", type=float, default=1, help="discriminator loss weight")
parser.add_argument("--critic_updates", type=int, default=3, help="Number of discriminator training")
parser.add_argument("--save_images", default='./experiments/drgan/imgs/', help="where to store images")
parser.add_argument("--save_models", default='./experiments/drgan/weights/', help="where to save models")
parser.add_argument("--gpu", type=str, default="2", help="gpu number")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.save_images, exist_ok=True)
os.makedirs(opt.save_models, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
def train_drgan():
   
    # construct models
    g = generator()
    d = discriminator()
    gan = DRGAN(g, d)
    
    # set up optimizer
    d_opt = Adam(lr = opt.lr_g, beta_1 = opt.b1, beta_2 = opt.b2, epsilon = opt.ep)
    d_on_g_opt = Adam(lr = opt.lr_d, beta_1 = opt.b1, beta_2 = opt.b2, epsilon = opt.ep)
    
    # compile models
    d.trainable = True
    d.compile(optimizer = d_opt, loss = wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [opt.lambda_gen, opt.lambda_dis]
    gan.compile(optimizer = d_on_g_opt, loss = loss, loss_weights = loss_weights)
    d.trainable = True
    
    # labels for D
    output_true_batch, output_false_batch = np.ones((opt.batch_size, 1)), np.zeros((opt.batch_size, 1))

    # load test images
    x_test, y_test = get_data(opt.test_num, opt.test_path)
    
    # train
    for epoch in range(opt.epoch_num):
        print('epoch: {}/{}'.format(epoch, opt.epoch_num))
        
        permutated_indexes_test = np.random.permutation(opt.test_num)
        gan_loss = 0
        
        for batch_i, (x_train_batch, y_train_batch) in enumerate(load_batch(opt.train_num, opt.batch_size, opt.train_path)):
            
            generated_images = g.predict(x = x_train_batch, batch_size = opt.batch_size)
            
            for _ in range(opt.critic_updates):
                d_loss_real = d.train_on_batch(y_train_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
            
            d.trainable = False
            gan_loss = gan.train_on_batch(x_train_batch, [y_train_batch, output_true_batch])
            d.trainable = True
            
        print("gan_loss: ", gan_loss)
        
        if ((epoch + 1) % opt.save_interval == 0):

            # save weights
            save_all_weights(d, g, opt.save_models, epoch)

            # save image
            batch_indexes_test = permutated_indexes_test[0 : 16]
            x_test_batch = x_test[batch_indexes_test]
            y_test_batch = y_test[batch_indexes_test]
            
            generated_images_test = g.predict(x = x_test_batch, batch_size = 16)
            
            plot_images(x_test_batch, True, opt.save_images, "test_src", step = epoch)
            plot_images(y_test_batch, True, opt.save_images, "test_gt", step = epoch)
            plot_images(generated_images_test, True, opt.save_images, "test_pre", step = epoch)
            
            # quantitative evaluation
            each_test_num = 16
            sum_ssim_test = 0
            sum_psnr_test = 0
            for i in range(each_test_num):
                
                src_img_test = y_test_batch[i]
                rec_str_test = generated_images_test[i]
                
                rec_img_test = np2img(rec_str_test)
                src_img_test = np2img(src_img_test)
                
                sum_ssim_test += ssim(src_img_test, rec_img_test, multichannel = True)
                sum_psnr_test += psnr(src_img_test, rec_img_test)
            
            test_ssim = sum_ssim_test / each_test_num
            test_psnr = sum_psnr_test / each_test_num
              
            print("test ssim: ", test_ssim)
            print("test psnr: ", test_psnr) 
            
            f = open(opt.save_images + 'eva.txt', 'a')
            f.write('Epoch:' + str(epoch) + '\n')
            f.write('Loss:' + str(gan_loss) + '\n')
            f.write('test ssim:' + str(test_ssim) + '\n')
            f.write('test psnr:' + str(test_psnr) + '\n')
            f.close()
            
    
if __name__ == '__main__':
    train_drgan()