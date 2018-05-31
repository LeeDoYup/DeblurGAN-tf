from __future__ import print_function

import os
import sys
import logging
import js

import tensorflow as tf 
import numpy as np 
from models.cgan_model import cgan

def linear_decay(initial=0.0001, step, start_step=150, end_step=300):
    '''
    return decayed learning rate
    It becomes 0 at end_step
    '''
    decay_period = end_step - start_step
    step_decay = (initial-0.0)/decay_period
    update_step = max(0, step-start_step)
    current_value = max(0, initial - (update_step)*step_decay)
    return current_value

def build_model(args):
    sess = tf.Session()
    model = cgan(sess, args)
    model.build_model()

def ready_batch_data():
    #return dataset (batch or total)
    pass

def main(args):
    config = json.load(open(args.config), 'r')
    #assert batch_num == 
    #assume there is a batch data pair:
    dataset = data_loader.load_data(args.data_path) #have te be developed
    #dataset = [num_dataset, ]
    num_dataset = len(dataset)
    num_batch = num_dataset/args.batch_num
    sess = tf.Session()
    model = cgan(sess,args)
    model.build_model()


    for iter in range(args.epoch):
        batch_loss_G, batch_loss_D = 0.0 ,0.0
        for i, data in enumerate(dataset):
            learning_rate = linear_decay(0.0001, iter)
            start_time = time.time()
            feed_dict_D = {model.input['blur_img']: data.blur_img,
                        model.input['real_img']: data.real_img,
                        model.learning_rate: learning_rate}

            for j in range(args.iter_gen):
                loss_G, adv_loss, perceptual_loss = 
                    model.run_optim_G(feed_dict=feed_dict_G, with_loss=True)
                batch_loss_G +=loss_G
                #logging: time, loss

            for j in range(args.iter_gen):
                loss_D = model.run_optim_G(feed_dict=feed_dict_D, with_loss=True)
                batch_loss_D +=loss_D
                #logging: time, loss

            batch_time = time.time() - start_time
            #logging

        batch_loss_G /= num_batch
        batch_loss_D /= num_batch
        #logging



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.parse_args('--is_training', action='store_true')
    parser.add_argument('-c', '--conf', type=str, default='configs/ilsvrc_sample.json')
    parser.add_argument('--iter_gen', type=int, default=5)
    parser.add_argument('--iter_disc', type=int, default=1)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='data/GOPRO_Large/train')

    '''
    currnet_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('-n', '--name', type=str, default='train')
    parser.add_argument('--checkpoint-dir', type=str, default=currnet_path+'/checkpoints/')
    parser.add_argument('--summary-dir', type=str, default=currnet_path+'/summaries/')
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
    #main(args)



