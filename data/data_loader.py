from __future__ import print_function

import os
import numpy as np
import cv2
import glob
import logging


def read_data_path(data_path, name='GOPRO', image_type='png'):
    dir_list = [dir for dir in glob.glob(data_path+'/*') if os.path.isdir(dir)]
    image_pair_path = []
    for i, dir in enumerate(dir_list):
        if not name in dir:
            dir_list.remove(dir)
            
    dir_image_pair(dir_list[0])

    for i, dir in enumerate(dir_list):
        image_pair_path.extend(dir_image_pair(dir))
    return image_pair_path


def dir_image_pair(dir_path, image_type='png'):
    blur_path = os.path.join(dir_path, 'blur')
    real_path = os.path.join(dir_path, 'sharp')

    blur_image_pathes = glob.glob(blur_path+'/*.'+image_type)
    real_image_pathes = glob.glob(real_path+'/*.'+image_type)
    assert len(blur_image_pathes) == len(real_image_pathes)
    pair_path = zip(blur_image_pathes, real_image_pathes)
    iter_pair_path = pair_path #for iteration
    for blur, real in iter_pair_path:
        name1=blur.split('/')[-1]
        name2=blur.split('/')[-1]
        if name1 != name2:
            pair_path.remove((blur, real))
            print("blur: %s, real: %s pair was removed in training data"%(name1, name2))
            #logging
    return pair_path


def read_image_pair(pair_path, resize_or_crop=None, image_size=(256,256)):
    image_blur = cv2.imread(pair_path[0], cv2.IMREAD_COLOR)
    image_blur = image_blur / 255.0
    image_real = cv2.imread(pair_path[1], cv2.IMREAD_COLOR)
    image_real = image_real / 255.0

    if resize_or_crop != None: 
        assert image_size != None

    if resize_or_crop == 'resize':
        image_blur = cv2.resize(image_blur, image_size, interpolation=cv2.INTER_AREA)
        image_real = cv2.resize(image_real, image_size, interpolation=cv2.INTER_AREA)

    if resize_or_crop == 'crop':
        image_blur = cv2.crop(image_blur, image_size)
        image_real = cv2.crop(image_real, image_size)

    return image_blur, image_real


if __name__ == '__main__':
    pair_path = read_data_path('/Users/kakaobrain/GOPRO_Large/train', name='GOPRO')
    image1, image2 = read_image_pair(pair_path[0], resize_or_crop='resize')

    
    cv2.imshow('image1',image1)
    cv2.imshow('image2',image2)
    cv2.waitKey(0)
