from __future__ import print_function

import numpy as np 
import tensorflow as tf
import os
import time
import logging 

from models.ops import * #discriminator, generator
#from base_model import BaseModel
from models.losses import *

image_shape = [None, 256,256,3]


class cgan(object):
    def name(self):
        return 'cgan'
    def __init__(self, sess, args):
        #BaseModel.initialize(opt)
        self.args = args
        self.sess = sess
        #self.global_

    def create_input_placeholder(self):
        self.input = {'blur_img': tf.placeholder(dtype=tf.float32, shape=image_shape, name='blur_img'),
            'real_img': tf.placeholder(dtype=tf.float32, shape=image_shape, name='real_img'),
            'gen_img': tf.placeholder(dtype=tf.float32, shape=image_shape, name='gen_img')
            }
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        print("[*] Placeholders are created")
        #logging


    def build_model(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.create_input_placeholder()
        self.G = generator(self.input['blur_img'])

        #if test mode, only generator is used.
        if self.args.is_training:
            self.D = discriminator(self.input['gen_img'])
            self.D_ = discriminator(self.input['real_img'])
            self.create_loss()
            t_vars = tf.trainable_variables() 
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
            self.d_vars = [var for var in t_vars if 'disc' in var.name]

            self.optim_G = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_G, var_list=self.g_vars)
            self.optim_D = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_D, var_list=self.d_vars)

        self.saver = tf.train.Saver()
        print("[*] C_GAN model build was completed")
        #loggi
        self.sess.run(tf.global_variables_initializer())
        print('hello')

    def run_optim_G(self, feed_dict, with_loss=True, with_out=False):
        _, loss_G, adv_loss, perceptual_loss, G_out = self.sess.run(
            [self.optim_G, self.loss_G, self.adv_loss, self.perceptual_loss, self.G],
            feed_dict=feed_dict)
        print("[*] Generator is optimized")
        #logging

        if with_loss and with_out:
            return loss_G, adv_loss, perceptual_loss, G_out
        elif with_loss:
            return loss_G, adv_loss, perceptual_loss
        elif with_out:
            G_out
        else:
            return
    def G_output(self, feed_dict):
        return self.sess.run(self.G, feed_dict=feed_dict)

    def run_optim_D(self, feed_dict, with_loss=True):
        _, loss_D= self.sess.run([self.optim_D, self.loss_D],
            feed_dict=feed_dict)
        print("[*] Discriminator is optimized")
        #logging

        if with_loss:
            return loss_D
        else:
            return


    def create_loss(self, regularizer = 100.):
        self.adv_loss = adv_loss(self.D)
        self.perceptual_loss = perceptual_loss(self.G, self.input['real_img']) #vgg19 feature have to be calculated
        
        self.loss_G = self.adv_loss + regularizer * self.perceptual_loss
        self.loss_D = wasserstein_loss(self.D, self.D_)
        print(" [*] loss functions are created")
        #logging


    def save_weights(self, checkpoint_dir, step):
        model_name = "DeblurGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def load_weights(self, checkpoint_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_nmae = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Success to read {}".format(ckpt_name))
            return False, 0



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('-c', '--conf', type=str, default='configs/config.json')
    parser.add_argument('--iter_gen', type=int, default=5)
    parser.add_argument('--iter_disc', type=int, default=1)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='data/GOPRO_Large/train')
    parser.add_argument('--data_name', type=str, default='GOPRO')

    parser.add_argument('--checkpoint_dir', type=str, default=currnet_path+'/checkpoints/')
    parser.add_argument('--summary_dir', type=str, default=currnet_path+'/summaries/')
    parser.add_argument('--data_name', type=str, default='GOPRO')

    parser.add_argument('--resize_or_crop', type=str, default='resize')
    parser.add_argument('--img_x', type=int, default=256)
    parser.add_argument('--img_y', type=int, default=256)

    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resize_or_crop', action='store_true')

    args = parser.parse_args()
    sess = tf.Session()
    test_cgan = cgan(sess)
    test_cgan.build_model()
    print(test_cgan.input)
    print(test_cgan.D)
    print(test_cgan.G)
    print(test_cgan.adv_loss)
    #print(test_cgan.perceptual_loss)
    #print(test_cgan.loss_G)
    print(test_cgan.loss_D)


