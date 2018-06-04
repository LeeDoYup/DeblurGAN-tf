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
    #assert batch_num == 
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
            print(iter, 'epoch,  ', i, 'batch, learning_rate',learning_rate)
            blur_img, real_img = loader.read_image_pair(data, 
                                    resize_or_crop = args.resize_or_crop, 
                                    image_size=(args.img_x, args.img_y))
            start_time = time.time()
            print("[!] Generator Optimization Start")
            for j in range(args.iter_gen):
                feed_dict_G = {model.input['blur_img']: blur_img,
                        model.input['real_img']: real_img,
                        model.learning_rate: learning_rate}
                
                loss_G, adv_loss, perceptual_loss, G_out = model.run_optim_G(feed_dict=feed_dict_G, 
                                                                with_loss=True, with_out=True)
                print(iter, 'epoch,  ', i, 'batch, Generator Loss: ', loss_G,
                        'adv loss: ', adv_loss, 'perceptual_loss: ', perceptual_loss)
                batch_loss_G +=loss_G
                #logging: time, loss

            feed_dict_D = {model.input['real_img']: real_img}
            D_ = model.D__output(feed_dict=feed_dict_D)
            
            feed_dict_D = {model.input['gen_img']: G_out,
                        model.input['real_img']: real_img,
                        model.input['y']: D_,
                        model.learning_rate: learning_rate}

            print("[!] Discriminator Optimization Start")
            for j in range(args.iter_disc):
                loss_D = model.run_optim_D(feed_dict=feed_dict_D, with_loss=True)
                batch_loss_D +=loss_D
                #logging: time, loss
                
                print(iter, 'epoch,  ', i, 'batch, Discriminator  Loss: ', loss_D)
            batch_time = time.time() - start_time
            print("Batch training time: ", batch_time)
            #logging

        batch_loss_G = batch_loss_G /(num_batch * args.iter_gen)
        batch_loss_D = batch_loss_D /(num_batch * args.iter_disc)
        print(iter, "th Batch Loss of G: ", batch_loss_G)
        print(iter, "th Batch Loss of D: ", batch_loss_D)
        #logging

        #if iter+1 % 30 == 0:        
            model.save_weights(args.checkpoint_dir, iter+1)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('-c', '--conf', type=str, default='configs/config.json')
    parser.add_argument('--iter_gen', type=int, default=5)
    parser.add_argument('--iter_disc', type=int, default=1)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--data_path', type=str, default='/data/private/data/GOPRO_Large/train/')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--model_dir', type=str, default='./checkpoints/')
    parser.add_argument('--summary_dir', type=str, default='./summaries/')
    parser.add_argument('--data_name', type=str, default='GOPRO')

    parser.add_argument('--resize_or_crop', type=str, default='resize')
    parser.add_argument('--img_x', type=int, default=256)
    parser.add_argument('--img_y', type=int, default=256)

    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--debug', action='store_true')

    '''
    currnet_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('-n', '--name', type=str, default='train')
    
    
    parser.add_argument('--debug', action='store_true')
    parser.parse_args('--is_training', action='store_true')
    '''
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    logging.getLogger("cgan.*").setLevel(level)

    
    '''
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
    args.summary_dir = os.path.join(args.summary_dir, args.name)

    for dirname in [args.checkpoint_dir, args.summary_dir]:
        if os.path.exists(dirname):
            if args.name == 'test':
                shutil.rmtree(dirname)
                logging.warning('%s directory is exists FORCE DELETED!', dirname)
            else:
                logging.error('%s directory is exists', dirname)
                if args.force is False:
                    exit(-1)
    '''
    main(args)



