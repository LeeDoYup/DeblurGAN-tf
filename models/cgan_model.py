from __future__ import print_function

import numpy as np 
import tensorflow as tf
import os
import time
import logging 

from ops import * #discriminator, generator
#from base_model import BaseModel
from losses import *

image_shape = [None, 256,256,3]


class cgan(object):
    def name(self):
        return 'cgan'
    def __init__(self, sess, args=args):
        #BaseModel.initialize(opt)
        self.args = args
        self.sess = sess
        self.global_

    def create_input_placeholder(self):
        self.input = {'blur_img': tf.placeholder(dtype=tf.float32, shape=image_shape),
            'real_img': tf.placeholder(dtype=tf.float32, shape=image_shape),
            'gen_img': tf.placeholder(dtype=tf.float32, shape=image_shape)
            }
        self.learning_rate = tf.placeholder(dtype=tf.float32)

    def build_model(self):
        self.saver = tf.train.Saver()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.create_input_placeholder()
        self.G = generator(self.input['blur_img'])

        #if test mode, only generator is used.
        if self.args.is_training:
            self.D = discriminator(self.input['gen_img'])
            self.create_loss()
            self.optim_g = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_G)
            self.optim_d = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_D)


    def create_loss(self, regularizer = 100):
        self.adv_loss = adv_loss(self.D)
        self.perceptual_loss = perceptual_loss(self.input['gen_img'], self.input['real_img']) #vgg19 feature have to be calculated
        
        self.loss_G = self.adv_loss + regularizer * self.perceptual_loss
        self.loss_D = wasserstein_loss(self.input['gen_img'], self.input['real_img'])


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
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Success to read {}".format(ckpt_name))
            return False, 0




if __name__ == '__main__':
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


