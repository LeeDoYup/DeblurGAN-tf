import tensorflow as tf

image_shape = (224,224,3)

def adv_loss(sigm):
  '''
  args: shape = [batch_size, 1]: it means the discriminator predict the sample is real.
  '''
  loss = tf.reduce_sum(-1*sigm)
  return loss

def perceptual_loss(gen_img, real_img):
  '''
  It have to input vgg feature of generated image & real_img_feat
  '''
  vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  loss_model = tf.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer('block3_conv3').output)
  loss_model.trainable=False
  #average with width, height
  loss =  tf.reduce_mean(tf.square(loss_model(gen_img)-loss_model(real_img)),axis=[1,2])
  
  #sum with channel
  loss = tf.reduce_sum(loss, axis=-1)
  return loss

def l2_loss(gen_img, real_img):
  return tf.reduce_mean(tf.abs(gen_img-real_img))


def wasserstein_loss(gen_img, real_img):
  assert gen_img.shape.as_list() == real_img.shape.as_list()
  return tf.reduce_mean(real_img*gen_img)