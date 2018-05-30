import numpy as np 
import tensorflow as tf
import os
import time
import logging 

from .ops import * #discriminator, generator
from .base_model import BaseModel
from .losses import *

class cgan(BaseModel):
    def name(self):
        return 'cgan'
    def __init__(self, opt):
        BaseModel.initialize(opt)

    def create_input_placeholder(self, input):
        self.input = {'blur_img': tf.placeholder(dtype=tf.float32, shape=None),
            'real_img': tf.placeholder(dtype=tf.float32, shape=None),
            'gen_img': tf.placeholder(dtype=tf.float32, shape=None)
            }

        self.label = {'real_img': tf.placeholder(dtype=tf.int32, shape=None),
            'gen_img': tf.placeholder(dtype=tf.int32, shape=None)}

    def build_model(self):
        self.D = discriminator(self.input['gen_img'])
        self.G = generator(self.input['blur_img'])

    def create_loss(self, regularizer = 100):
        self.adv_loss = adv_loss(self.D)
        self.perceptual_loss = perceptual_loss([self.input['blur_img'], self.input['gen_img']]) #vgg19 feature have to be calculated
        
        self.loss_G = self.adv_loss + regularizer * self.perceptual_loss
        self.loss_D = wasserstein_loss(self.input['gen_img'], self.input['real_img'])

