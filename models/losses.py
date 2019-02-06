import tensorflow as tf
from models.vgg_model import VGG

vgg_model = VGG('vgg19')

def adv_loss(sigm, name='adv_loss'):
  '''
  args: shape = [batch_size, 1]: it means the discriminator predict the sample is real.
  '''
  with tf.name_scope(name=name) as scope:
    loss_fake = sigm[0]
    return tf.reduce_mean(-1.0 * loss_fake)

def perceptual_loss(gen_img, real_img, name='perceptual_loss'):
  with tf.name_scope(name=name) as scope:
    real_feat = vgg_model.model(real_img)
    gen_feat = vgg_model.model(gen_img)
    loss =  tf.reduce_mean(tf.square(gen_feat - real_feat))
    return loss

def l1_loss(gen_input, real_input, name='L1_loss'):
  with tf.name_scope(name=name) as scope:
    return tf.reduce_mean(tf.abs(gen_input-real_input))

def wasserstein_loss(gen_prob, real_prob, with_gp=True, d_x_hat=None, name='wasserstein_loss'):
  with tf.name_scope(name=name) as scope:
    loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_prob+1e-8, labels=tf.zeros_like(gen_prob))
    loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_prob+1e-8, labels=tf.ones_like(real_prob))

    loss_D = loss_fake + loss_real
    return tf.reduce_sum(loss_D)

def wasserstein_gp_loss(prob, gt, d_gp, x_hat, name='wasserstein_gp_loss'):
  with tf.name_scope(name=name) as scope:
    loss_D = tf.reduce_mean(tf.multiply(prob, gt))
    grad_d_x_hat = tf.gradients(d_gp, [x_hat])[0]
    red_idx = list(range(1, x_hat.shape.ndims))

    slopes = tf.sqrt(1e-8+tf.reduce_sum(tf.square(grad_d_x_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))

    return loss_D, 10.0 * gradient_penalty
