import tensorflow as tf
from models.vgg_model import VGG

image_shape = (256,256,3)
vgg_model = VGG('vgg19')

def adv_loss(sigm):
  '''
  args: shape = [batch_size, 1]: it means the discriminator predict the sample is real.
  '''
  loss = tf.reduce_sum(-1*sigm)
  return loss

def perceptual_loss(gen_img, real_img):
  gen_feat, real_feat = vgg_model.get_pair_feature(gen_img, real_img)
  loss =  tf.reduce_mean(tf.square(gen_feat - real_feat),axis=[1,2])
  #sum with channel
  loss = tf.reduce_sum(loss, axis=-1)
  return loss

def l2_loss(gen_img, real_img):
  return tf.reduce_mean(tf.abs(gen_img-real_img))


def wasserstein_loss(gen_prob, real_prob):
  return tf.reduce_mean(real_prob*gen_prob)
