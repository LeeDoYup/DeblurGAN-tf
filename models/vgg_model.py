from __future__ import print_function

import tensorflow as tf 
import numpy as np
import logging

class VGG(object):
    def __init__(self, name, include_top=False, weights='imagenet'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            if name.upper() == 'VGG19':
                self.vgg = tf.keras.applications.VGG19(include_top=include_top,
                                weights=weights)
            elif name.upper() == 'VGG16':
                self.vgg = tf.keras.applications.VGG16(include_top=include_top,
                                weights=weights)
            else:
                raise TypeError('Not supported model: VGG{}'.format(name))

            self.model = tf.keras.Model(inputs=self.vgg.input,
                                outputs = self.vgg.get_layer('block3_conv3').output)
            self.model.trainable=False
            print(" [*] ", name, " model was created")

    def get_pair_feature(self, gen_img, real_img):
        assert gen_img.shape.as_list() == real_img.shape.as_list()
        batch_num = gen_img.shape.as_list()[0]

        pair = tf.concat([gen_img, real_img], axis=0)
        output = self.model(pair)
        gen_feat, real_feat = output[:batch_num,:,:,:], output[batch_num:,:,:,:]
        return gen_feat, real_feat

if __name__=='__main__':
    model = VGG('vgg19')
    vars = tf.trainable_variables()
    for i, var in enumerate(vars):
        print(i,"-th variable: ", var)

    print(model.get_feature(np.ones([1,256,256,3])))




