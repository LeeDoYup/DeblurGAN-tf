from __future__ import print_function

import numpy as np 
import tensorflow as tf
import os
import time
import logging 

from models.ops import * #discriminator, generator
#from base_model import BaseModel
from models.losses import *


class cgan(object):
    def name(self):
        return 'cgan'

    def __init__(self, sess, args):
        self.args = args
        self.sess = sess
        self.image_op = []
        self.image_shape = [None, args.img_h, args.img_w, args.img_c]

    def create_input_placeholder(self):
        self.input = {'blur_img': tf.placeholder(dtype=tf.float32, shape=self.image_shape, name='blur_img'),
            'real_img': tf.placeholder(dtype=tf.float32, shape=self.image_shape, name='real_img')
            }
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        print("[*] Placeholders are created")


    def build_model(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.create_input_placeholder()
        self.G = generator(self.input['blur_img'])

        #if custom-test mode, only generator is used.
        try:
            if self.args.is_test:
                return
        except Exception as e:
            print("[*] Training model creation start")

        self.D = discriminator(tf.concat([self.G, self.input['real_img']], axis=0))
        self.gt = tf.concat([tf.ones([self.args.batch_num, 1]), -1.0 * tf.ones([self.args.batch_num,1])], axis=0)
        self.x_hat = get_x_hat(self.G, self.input['real_img'], self.args.batch_num)
        self.D_gp = discriminator(self.x_hat)
        
        self.create_loss()
        
        with tf.name_scope('optimizer') as scope:
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
        if self.args.tf_image_monitor:        
            image_op = [tf.summary.image('GOPRO/blur_img', self.input['blur_img']),
                        tf.summary.image('GOPRO/real_img', self.input['real_img']),
                        tf.summary.image('GOPRO/pred_img', self.G)]
            self.image_summary_op = tf.summary.merge(image_op)
        

    def run_optim_G(self, feed_dict):
        summary, _, loss_G, adv_loss, perceptual_loss, step= self.sess.run(
            [self.gen_summary_op, self.optim_G, self.loss_G, self.adv_loss, self.perceptual_loss, self.global_step],
            feed_dict=feed_dict)

        self.writer.add_summary(summary, step)
        return loss_G, adv_loss, perceptual_loss

    def G_output(self, feed_dict):
        return self.sess.run(self.G, feed_dict=feed_dict)

    def D_output(self, feed_dict):
        return self.sess.run(self.D, feed_dict=feed_dict)
    
    def run_optim_D(self, feed_dict, with_image=False):
        if with_image:
            fetch = [self.optim_D, self.disc_summary_op, self.image_summary_op,  self.loss_D, self.loss_disc, self.loss_gp, self.global_step]
        else:
            fetch = [self.optim_D, self.disc_summary_op, self.loss_D, self.loss_disc, self.loss_gp, self.global_step]

        run_list = self.sess.run(fetch, feed_dict=feed_dict)

        self.writer.add_summary(run_list[1], run_list[-1])
        if with_image:
            self.writer.add_summary(run_list[2])
        return run_list[-4], run_list[-3], run_list[-2]

    def create_loss(self, regularizer = 100.):
        with tf.name_scope('loss_G') as scope:
            self.adv_loss = adv_loss(self.D)
            self.perceptual_loss = perceptual_loss(self.G, self.input['real_img']) 
            self.loss_G = self.adv_loss + regularizer * self.perceptual_loss
        
        gen_summary = [tf.summary.scalar('loss_G/loss_G', self.loss_G),
                            tf.summary.scalar('loss_G/adversarial_loss', self.adv_loss),
                            tf.summary.scalar('loss_G/contents_loss', self.perceptual_loss)]
        self.gen_summary_op = tf.summary.merge(gen_summary)

        with tf.name_scope('loss_D') as scope:
            self.loss_disc, self.loss_gp = wasserstein_gp_loss(self.D, self.gt, self.D_gp, self.x_hat)
            self.loss_D = self.loss_disc + self.loss_gp

        disc_summary = [tf.summary.scalar('loss_D/loss_D', self.loss_D),
                            tf.summary.scalar('loss_D/disc_loss', self.loss_disc),
                            tf.summary.scalar('loss_D/gradient_penalty', self.loss_gp)]
        self.disc_summary_op = tf.summary.merge(disc_summary)

        print(" [*] loss functions are created")

    def save_weights(self, checkpoint_dir, step):
        model_name = self.args.model_name #"DeblurGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step,
                        write_meta_graph=False)

    def load_weights(self, checkpoint_dir):
        import re
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            try:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                print(" [*] Success to read {}".format(ckpt_name))
                return True
            except Exception as e:
                print("[!] Can't load: ", ckpt_name)
        else:
            print("[*] There is no saved file.")
            print("[*] Training Start from the scratch.")
            return False



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('-c', '--conf', type=str, default='configs/config.json')
    parser.add_argument('--iter_gen', type=int, default=5)
    parser.add_argument('--iter_disc', type=int, default=1)
    parser.add_argument('--batch_num', type=int, default=1)

    parser.add_argument('--checkpoint_dir', type=str, default=currnet_path+'/checkpoints/')
    parser.add_argument('--model_name', type=str, default='DeblurGAN.model')
    parser.add_argument('--summary_dir', type=str, default=currnet_path+'/summaries/')
    parser.add_argument('--data_name', type=str, default='GOPRO')

    parser.add_argument('--resize_or_crop', type=str, default='resize')
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_c', type=int, default=3)

    parser.add_argument('--is_test', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    sess = tf.Session()
    test_cgan = cgan(sess)
    test_cgan.build_model()
    print(test_cgan.input)
    print(test_cgan.D)
    print(test_cgan.G)
    print(test_cgan.adv_loss)
    print(test_cgan.loss_D)


