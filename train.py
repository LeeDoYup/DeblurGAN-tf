from __future__ import print_function

import time
import os
import sys
import logging
import json

import tensorflow as tf 
import numpy as np 

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

def main(args):
    #config = json.load(open(args.conf), 'r')
    #assume there is a batch data pair:

    #dataset = data_loader.load_data(args.data_path) #have te be developed
    dataset = loader.read_data_path(args.data_path, name=args.data_name)
    #dataset = [num_dataset, ]
    num_dataset = len(dataset)
    num_batch = num_dataset/args.batch_num
    sess = tf.Session()
    model = cgan(sess, args)
    model.build_model()
    model.sess.run(tf.global_variables_initializer())
    
    for iter in range(args.epoch):
        batch_loss_G, batch_loss_D = 0.0 ,0.0
        for i, data in enumerate(dataset):
            learning_rate = linear_decay(0.0001, iter)
            blur_img, real_img = loader.read_image_pair(data, 
                                    resize_or_crop = args.resize_or_crop, 
                                    image_size=(args.img_x, args.img_y))

            start_time = time.time()
            
            logging.info("[!] Generator Optimization Start")
            #for j in range(args.iter_gen):
            feed_dict = {model.input['blur_img']: blur_img,\
                        model.input['real_img']: real_img,\
                        model.learning_rate: learning_rate}

            if i % 5 == 0 :    
                loss_G, adv_loss, perceptual_loss = model.run_optim_G(feed_dict=feed_dict, 
                                                                with_loss=True)
                logging.info('%d epoch,  %d batch, Generator Loss:  %f, add loss: %f, perceptual_loss: %f',\
                             iter, i, loss_G, adv_loss, perceptual_loss)
                batch_loss_G +=loss_G
            
            #Ready for Training Discriminator
            logging.info("[!] Discriminator Optimization Start")
            
            #for j in range(args.iter_disc):
            loss_D = model.run_optim_D(feed_dict=feed_dict_D, with_loss=True)
            print(loss_D)
            batch_loss_D += loss_D
            logging.info('%d epoch,  %d  batch, Discriminator  Loss:  %f', iter, i, loss_D)

            batch_time = time.time() - start_time
            print("Time for training a batch: ", batch_time)
            
        batch_loss_G = batch_loss_G /(num_batch / args.iter_gen)
        batch_loss_D = batch_loss_D /(num_batch / args.iter_disc)
        logging.info("%d iter's Average Batch Loss:: G_Loss: %f, D_Loss: %f", iter, batch_loss_G, batch_loss_D)

        if (iter+1) % 30 == 0 or iter == (args.epoch-1):        
            model.save_weights(args.checkpoint_dir, iter+1)
    logging.info("[!] test started") 
    dataset = loader.read_data_path(args.data_path, name=args.data_name)
    
    for i, data in enumerate(dataset):
        if os.path.exists('./test_result'):
            os.mkdir('./test_result')
        blur_img, real_img = loader.read_image_pair(data, resize_or_crop = args.resize_or_crop,
                    image_size=(args.img_x, args.img_y))
        feed_dict_G = {model.input['blur_img']: blur_img}
        G_out = model.G_output(feed_dict=feed_dict_G)
        cv2.imwrite('./test_result/'+str(i)+'_blur.png', blur_img)
        cv2.imwrite('./test_result/'+str(i)+'_real.png', real_img)
        cv2.imwrite('./test_result/'+str(i)+'_gen.png', G_out)

    logging.info("[*] test done")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('-c', '--conf', type=str, default='configs/config.json')
    parser.add_argument('--iter_gen', type=int, default=1)
    parser.add_argument('--iter_disc', type=int, default=5)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--data_path', type=str, default='/data/private/data/GOPRO_Large/train/')
    parser.add_argument('--data_path_t', type=str, default='/data/private/data/GOPRO_Large/test/')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--model_name', type=str, default='DeblurGAN.model')
    parser.add_argument('--summary_dir', type=str, default='./summaries/')
    parser.add_argument('--data_name', type=str, default='GOPRO')

    parser.add_argument('--resize_or_crop', type=str, default='resize')
    parser.add_argument('--img_x', type=int, default=256)
    parser.add_argument('--img_y', type=int, default=256)

    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--debug', action='store_true')

     
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    logging.getLogger("cgan.*").setLevel(level)

    
    dataset = loader.read_data_path(args.data_path, name=args.data_name)
    main(args)



