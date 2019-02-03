from __future__ import print_function

import numpy as np 
import tensorflow as tf
import os
import time
import logging 

from models.ops import * #discriminator, generator
#from base_model import BaseModel
from models.losses import *

image_shape = [1,256,256,3]


class cgan(object):
    def name(self):
        return 'cgan'
    def __init__(self, sess, args):
        #BaseModel.initialize(opt)
        self.args = args
        self.sess = sess
        self.image_op = []

    def create_input_placeholder(self):
        self.input = {'blur_img': tf.placeholder(dtype=tf.float32, shape=image_shape, name='blur_img'),
            'real_img': tf.placeholder(dtype=tf.float32, shape=image_shape, name='real_img')
            }
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        print("[*] Placeholders are created")


    def build_model(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.create_input_placeholder()
        self.G = generator(self.input['blur_img'])

        #if test mode, only generator is used.
        if self.args.is_training:
            self.D = discriminator(tf.concat([self.G, self.input['real_img']], axis=0))
            self.gt = tf.concat([tf.zeros([self.args.batch_num, 1]), tf.ones([self.args.batch_num,1])], axis=0)
            self.x_hat = get_x_hat(self.G, self.input['real_img'], self.args.batch_num)
            self.D_gp = discriminator(self.x_hat)

            self.create_loss()

            t_vars = tf.trainable_variables() 
            self.g_vars = [var for var in t_vars if 'generator' in var.name]
            self.d_vars = [var for var in t_vars if 'disc' in var.name]

            self.optim_G = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_G, self.global_step, self.g_vars)
            self.optim_D = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_D, self.global_step, self.d_vars)
        
        self.saver = tf.train.Saver()
        print("[*] C_GAN model build was completed")
        self.writer = tf.summary.FileWriter(self.args.summary_dir, self.sess.graph)
        vars = (tf.trainable_variables())
        for var in vars: print(var)

        image_op = [tf.summary.image('GOPRO/blur_img', self.input['blur_img']),
                    tf.summary.image('GOPRO/real_img', self.input['real_img']),
                    tf.summary.image('GOPRO/pred_img', self.G)]
        self.image_summary_op = tf.summary.merge(image_op)

    def run_optim_G(self, feed_dict, with_loss=True, step=0):
        summary, _, loss_G, adv_loss, perceptual_loss, step= self.sess.run(
            [self.gen_summary_op, self.optim_G, self.loss_G, self.adv_loss, self.perceptual_loss, self.global_step],
            feed_dict=feed_dict)

        self.writer.add_summary(summary, step)
        if with_loss:
            return loss_G, adv_loss, perceptual_loss
        else:
            return

    def G_output(self, feed_dict):
        return self.sess.run(self.G, feed_dict=feed_dict)

    def D_output(self, feed_dict):
        return self.sess.run(self.D, feed_dict=feed_dict)
    
    def run_optim_D(self, feed_dict, with_loss=True, step=0):
        #D_ = self.D__output(feed_dict=feed_dict)
        summary, img_summary, _, loss_D, step = self.sess.run([self.disc_summary_op, self.image_summary_op,\
                                            self.optim_D, self.loss_D, self.global_step],
                                            feed_dict=feed_dict)
        self.writer.add_summary(summary, step)
        self.writer.add_summary(img_summary)

        if with_loss:
            return loss_D
        else:
            return

    def create_loss(self, regularizer = 100.):
        self.adv_loss = adv_loss(self.D)
        self.perceptual_loss = perceptual_loss(self.G, self.input['real_img']) #vgg19 feature have to be calculated
        
        self.loss_G = self.adv_loss + regularizer * self.perceptual_loss
        self.loss_D = wasserstein_gp_loss(self.D, self.gt,self.D_gp, self.x_hat)

        gen_summary = [tf.summary.scalar('loss/D/loss_D', self.loss_D)]
        disc_summary = [tf.summary.scalar('loss/G/loss_G', tf.reduce_mean(self.loss_G)),
                        tf.summary.scalar('loss/G/adv_loss', tf.reduce_mean(self.adv_loss)),
                        tf.summary.scalar('loss/G/perceptual_loss', tf.reduce_mean(self.perceptual_loss))]

        self.gen_summary_op = tf.summary.merge(gen_summary)
        self.disc_summary_op = tf.summary.merge(disc_summary)

        print(" [*] loss functions are created")

    def save_weights(self, checkpoint_dir, step):
        model_name = self.args.model_name #"DeblurGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                os.path.join(checkpoint_dir, model_name),
                global_step=step)

    def load_weights(self, checkpoint_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_nmae = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
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
    parser.add_argument('--model_name', type=str, default='DeblurGAN.model')
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


