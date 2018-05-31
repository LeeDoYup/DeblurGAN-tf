from __future__ import print_function

import time
import os
import sys
import logging
import js

import tesorflow as tf 
import numpy as np 
from model.cgan_model import cgan 

def build_model(args):
    sess = tf.Session()
    model = cgan()

def main(args):
    config = json.load(open(args.config), 'r')

    sess = tf.Session()
    model = cgan(sess, args)
    model.build_model()

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
    parser.add_argument('--data_path', type=str, default='data/GOPRO_Large/train')

    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    logging.getLogger("cgan.*").setLevel(level)

    main(args)