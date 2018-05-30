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
    def __init__(self, sess, opt=None):
        #BaseModel.initialize(opt)
        self.sess = sess

    def create_input_placeholder(self):
        self.input = {'blur_img': tf.placeholder(dtype=tf.float32, shape=image_shape),
            'real_img': tf.placeholder(dtype=tf.float32, shape=image_shape),
            'gen_img': tf.placeholder(dtype=tf.float32, shape=image_shape)
            }

    def build_model(self):
        self.create_input_placeholder()
        self.D = discriminator(self.input['gen_img'])
        self.G = generator(self.input['blur_img'])
        self.create_loss()


    def create_loss(self, regularizer = 100):
        self.adv_loss = adv_loss(self.D)
        #self.perceptual_loss = perceptual_loss(self.input['gen_img'], self.input['real_img']) #vgg19 feature have to be calculated
        
        #self.loss_G = self.adv_loss + regularizer * self.perceptual_loss
        self.loss_D = wasserstein_loss(self.input['gen_img'], self.input['real_img'])



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


