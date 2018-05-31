from __future__ import print_function

import os
import numpy 
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




if __name__ == '__main__':
    read_data_path('/Users/kakaobrain/GOPRO_Large/train', name='GOPRO')
