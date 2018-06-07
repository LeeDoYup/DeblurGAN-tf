from __future__ import print_function

import time
import os
import sys
import logging
import js

import tesorflow as tf 
import numpy as np 
from data.data_loader import *
from model.cgan_model import cgan 


def build_model(args):
    sess = tf.Session()
    model = cgan()

def main(args):
    config = json.load(open(args.config), 'r')

    sess = tf.Session()
    model = cgan(sess, args)
    model.build_model()
    model.load_weights(args.checkpoint_dir)
    
    dataset  = glob.glob(os.path.join(args.data_path_t, '*.'+args.img_type))
    
    
    if not os.path.exist(args.result_dir):
        os.mkdir(args.result_dir)

    for i, data in enumerate(dataset):
        logging.info("%s image deblur starts", %data)
        blur_img = read_image(data)
        logging.debug("%s image was loaded", %data)
        feed_dict_G = {model.input['blur_img']: blur_img}
        G_out = model.G_output(feed_dict=feed_dict_G)
        logging.debug("The image was converted")
        logging.deug(G_out)
        cv2.imwrite(os.path.join(args.result_dir, str(i)+'_blur.png'), blur_img)
        cv2.imwrite(os.path.join(args.result_dir, str(i)+'_result.png'), G_out)
        logging.info("Image save was completed")
    #load save checkpoint files

    #for i:end of blur image
    #run generator with blur image input
    #save result image
    #iterate until blur image end



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.parse_args('--is_training', action='store_true')
    parser.add_argument('-c', '--conf', type=str, default='configs/ilsvrc_sample.json')
    parser.add_argument('--iter_gen', type=int, default=5)
    parser.add_argument('--iter_disc', type=int, default=1)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='/data/private/data/GOPRO_Large/train/')
    parser.add_argument('--data_path_t', type=str, default='./test_data/')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--model_name', type=str, default='DeblurGAN.model')
    parser.add_argument('--summary_dir', type=str, default='./summaries/')
    parser.add_arguemnt('--data_name', type=str, default='GOPRO')
    parser.add_argument('--result_dir', type=str, default='./result_dir')
    parser.add_argument('--debug', action='store_true')
    parset.add_argument('--img_type', type=str, default='png')
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    logging.getLogger("cgan.*").setLevel(level)

    main(args)
