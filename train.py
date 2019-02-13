from __future__ import print_function

import time
import os
import sys
import logging
import json

import tensorflow as tf 
import numpy as np 
import cv2

import data.data_loader as loader
from models.cgan_model import cgan
from models.ops import *
os.system('http_proxy_on')

def linear_decay(initial=0.0001, step=0, start_step=150, end_step=300):
    '''
    return decayed learning rate
    It becomes 0 at end_step
    '''
    decay_period = end_step - start_step
    step_decay = (initial-0.0)/decay_period
    update_step = max(0, step-start_step)
    current_value = max(0, initial - (update_step)*step_decay)
    return current_value

def train(args):
    #assume there is a batch data pair:

    dataset = loader.read_data_path(args.data_path_train, name=args.data_name)
    num_dataset = len(dataset)
    num_batch = num_dataset/args.batch_num
    sess = tf.Session()
    model = cgan(sess, args)
    model.build_model()
    model.sess.run(tf.global_variables_initializer())
    model.load_weights(args.checkpoint_dir)
     
    for iter in range(args.epoch):
        learning_rate = linear_decay(0.0001, iter)
        for i, data in enumerate(dataset):
            blur_img, real_img = loader.read_image_pair(data, 
                                    resize_or_crop = args.resize_or_crop, 
                                    image_size=(args.img_h, args.img_w))

            feed_dict = {model.input['blur_img']: blur_img,\
                        model.input['real_img']: real_img,\
                        model.learning_rate: learning_rate}
            
               
            loss_G, adv_loss, perceptual_loss = model.run_optim_G(feed_dict=feed_dict) 
            logging.info('%d epoch,  %d batch, Generator Loss:  %f, add loss: %f, perceptual_loss: %f',\
                             iter, i, loss_G, adv_loss, perceptual_loss)

            #Ready for Training Discriminator
            for _ in range(args.iter_disc):
                loss_D, loss_disc, loss_gp  = model.run_optim_D(feed_dict=feed_dict, with_image=args.tf_image_monitor)
                
            logging.info('%d epoch,  %d  batch, Discriminator  Loss:  %f, loss_disc:  %f, gp_loss: %f', iter, i, loss_D, loss_disc, loss_gp)
            
        if (iter+1) % 50 == 0 or iter == (args.epoch-1):        
            model.save_weights(args.checkpoint_dir, model.global_step)
    
    logging.info("[!] test started") 
    dataset = loader.read_data_path(args.data_path_test, name=args.data_name)
    
    for i, data in enumerate(dataset):
        if not os.path.exists('./test_result'):
            os.mkdir('./test_result')
        blur_img, real_img = loader.read_image_pair(data, resize_or_crop = args.resize_or_crop,
                    image_size=(args.img_h, args.img_w))
        feed_dict_G = {model.input['blur_img']: blur_img}
        G_out = model.G_output(feed_dict=feed_dict_G)
        cv2.imwrite('./test_result/'+str(i)+'_blur.png', (blur_img[0]+1.0)/2.0 *255.)
        cv2.imwrite('./test_result/'+str(i)+'_real.png', (real_img[0]+1.0)/2.0 *255.)
        cv2.imwrite('./test_result/'+str(i)+'_gen.png', (G_out[0]+1.0)/2.0*255.)
        logging.info("Deblur Image is saved (%d/%d) ", i, len(dataset))
    logging.info("[*] test done")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--iter_gen', type=int, default=1)
    parser.add_argument('--iter_disc', type=int, default=5)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--data_path_train', type=str, default='/data/private/data/GOPRO_Large/train/')
    parser.add_argument('--data_path_test', type=str, default='/data/private/data/GOPRO_Large/test/')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--model_name', type=str, default='DeblurGAN.model')
    parser.add_argument('--summary_dir', type=str, default='./summaries/')
    parser.add_argument('--data_name', type=str, default='GOPRO')
    parser.add_argument('--tf_image_monitor', type=bool, default=False)

    parser.add_argument('--resize_or_crop', type=str, default='resize')
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_c', type=int, default=3)

    parser.add_argument('--debug', action='store_true')
     
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    logging.getLogger("DeblurGAN_TRAIN.*").setLevel(level)

    
    train(args)



