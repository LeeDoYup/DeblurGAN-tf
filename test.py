from __future__ import print_function

import time
import os
import sys
import logging
import js
import tesorflow as tf 
import numpy as np 
import cv2
from data.data_loader import *
from model.cgan_model import cgan 


def build_model(args):
    sess = tf.Session()
    model = cgan()

def test(args):
    sess = tf.Session()
    model = cgan(sess, args)
    model.build_model()
    model.sess.run(tf.global_variables_initializer())
    model.load_weights(args.checkpoint_dir)

    dataset = read_data_path_custom(args.data_path_test, image_type=args.imge_type)
    image_size = (args.img_h, args.img_w)
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for i, data in enumerate(dataset):
        logging.info("%s image deblur starts", data)
        blur_img = read_image(data, resize_or_crop=args.resize_or_crop, image_size=image_size)
        logging.debug("%s image was loaded", data)

        feed_dict_G = {model.input['blur_img']: blur_img}
        G_out = model.G_output(feed_dict=feed_dict_G)
        logging.debug("The image was converted")

        cv2.imwrite(os.path.join(args.result_dir, 'sharp_'+data.split('/')[-1]), (G_out[0]+1.0)/2.0*255.0)
        logging.info("%s Image save was completed", data)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path_test', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default='./result_dir')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--model_name', type=str, default='DeblurGAN.model')

    parser.add_argument('--img_type', type=str, default='png')
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=256)
    parser.add_argument('--img_c', type=int, default=3)


    parser.add_argument('--is_test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    logging.getLogger("DeblurGAN_TEST.*").setLevel(level)

    test(args)
