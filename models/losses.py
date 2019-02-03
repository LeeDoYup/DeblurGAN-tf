import tensorflow as tf
from models.vgg_model import VGG

image_shape = (256,256,3)
vgg_model = VGG('vgg19')

def adv_loss(sigm, name='loss/G/adv_loss'):
  '''
  args: shape = [batch_size, 1]: it means the discriminator predict the sample is real.
  '''
  with tf.name_scope(name=name) as scope:
    loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=sigm, labels = tf.zeros_like(sigm))
    return -1.0 * loss_fake

def perceptual_loss(gen_img, real_img, name='loss/G/perceptual_loss'):
  with tf.name_scope(name=name) as scope:
    real_feat = vgg_model.model(real_img)
    gen_feat = vgg_model.model(gen_img)
    loss =  tf.reduce_mean(tf.square(gen_feat - real_feat),axis=[1,2])
    loss = tf.reduce_sum(loss, axis=-1)
    return loss

def l2_loss(gen_img, real_img, name='loss/G/L2_loss'):
  with tf.name_scope(name=name) as scope:
    return tf.reduce_mean(tf.abs(gen_img-real_img))

def wasserstein_loss(gen_prob, real_prob, with_gp=True, d_x_hat=None, name='wasserstein_loss'):
  with tf.name_scope(name=name) as scope:
    loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_prob+1e-8, labels=tf.zeros_like(gen_prob))
    loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_prob+1e-8, labels=tf.ones_like(real_prob))

    loss_D = loss_fake + loss_real
    return tf.reduce_sum(loss_D)

def wasserstein_gp_loss(prob, gt, d_gp, x_hat, name='loss/D/wasserstein_gp_loss'):
  with tf.name_scope(name=name) as scope:
    loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prob, labels=gt))
    
    grad_d_x_hat = tf.gradients(d_gp, [x_hat])[0]
    red_idx = list(range(1, x_hat.shape.ndims))

    slopes = tf.sqrt(1e-8+tf.reduce_sum(tf.square(grad_d_x_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)

    return loss_D + 10.0 * gradient_penalty
