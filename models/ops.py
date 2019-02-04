import tensorflow as tf
import numpy as np 
import logging 

def leak_relu(x, leak=0.2, name="leak_relu"):
  return tf.maximum(x, leak*x)

def concat(tensors, axis, *args, **kwargs):
  return tf.concat(tensors, axis, *args, **kwargs)


def norm_layer(input, ntype='instance', **kargs):
  if ntype == 'instance':
    n_layer = tf.contrib.layers.instance_norm(input)
  elif ntype == 'batch':
    n_layer = tf.contrib.layers.batch_norm(input)
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % ntype)
  return n_layer


def conv2d(input_, output_dim, kernel_h=3, kernel_w=None, stride_h=1, stride_w=None, padding='SAME', reuse=False, initializer=None, use_bias = True, name="conv2d"):
  
  if kernel_w == None: kernel_w = kernel_h
  if stride_w == None: stride_w = stride_h
  if initializer == None: initializer = tf.contrib.layers.xavier_initializer()

  with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
    if reuse==True: scope.reuse_variables()
    w = tf.get_variable('w', [kernel_h, kernel_w, input_.get_shape()[-1], output_dim],
      initializer=initializer)
    conv = tf.nn.conv2d(input_, w, strides=[1,stride_h, stride_w, 1], padding=padding)

    if use_bias:
      b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
      conv = tf.nn.bias_add(conv, b)

    return conv

def deconv2d(input_, output_dim, kernel_h=3, kernel_w=None, stride_h=1, stride_w=None, padding='SAME', initializer=None, use_bias = True, name="deconv2d"):
  
  if kernel_w == None: kernel_w = kernel_h
  if stride_w == None: stride_w = stride_h
  if initializer == None: initializer = tf.contrib.layers.xavier_initializer()

  with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
    deconv = tf.layers.conv2d_transpose(input_, output_dim, [kernel_h, kernel_w], strides=[stride_h, stride_w], padding=padding, use_bias=use_bias)

  return deconv


def conv_block(x, nf, k, s, p='SAME', ntype=None, name='conv_block'):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
    x = conv2d(x, nf, kernel_h=k, stride_h=s, name='conv')
    if not ntype == None:
      x = norm_layer(x, ntype)
    x = leak_relu(x)
  return x


def res_block(input_, output_dim, name='res_block', is_dropout=False, drop_p=0.5):
  shortcut = input_
  num_input_c = shortcut.shape.as_list()[-1]

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    conv = conv2d(input_, output_dim, name=name+'/conv1')
    conv = norm_layer(conv, 'instance')
    conv = tf.nn.relu(conv)

    if is_dropout:
      conv = tf.nn.dropout(conv, keep_prob = drop_p)

    #conv = conv2d(conv, output_dim, num_input_c * 2, name=name+'/conv2')
    conv = conv2d(conv, output_dim,  name=name+'/conv2')
    conv = norm_layer(conv, 'instance')

    conv = tf.identity(conv+shortcut, name='residual_block_output')

  return conv


def fc_layer(input_, output_dim, initializer = None, activation='linear', name=None):
  if initializer == None: initializer = tf.contrib.layers.xavier_initializer()
  shape = input_.get_shape().as_list()
  with tf.variable_scope(name or "Linear", reuse=tf.AUTO_REUSE) as scope:
    if len(shape) > 2 : input_ = tf.layers.flatten(input_)
    shape = input_.get_shape().as_list()

    w = tf.get_variable("fc_w", [shape[1], output_dim], dtype=tf.float32, initializer = initializer)
    b = tf.get_variable("fc_b", [output_dim], initializer = tf.constant_initializer(0.0))

    result = tf.matmul(input_, w) + b

    if activation == 'linear':
      return result
    elif activation == 'relu':
      return tf.nn.relu(result)
    elif activation == 'sigmoid':
      return tf.nn.sigmoid(result)
    elif activation == 'tanh':
      return tf.nn.tanh(result)


def generator(input, ngf=64, num_block=9, ntype='instance'):
  x = input
  count_l = 0 #counter of layer for naming layer
  with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
    '''
    (downsampling) conv layer
    '''
    with tf.variable_scope('h'+str(count_l), reuse=tf.AUTO_REUSE) as scope:
      #reflectionpadding2d (3,3)
      x = conv2d(x, ngf, kernel_h=7, stride_h=1, padding='SAME', name='g_0_conv')#VaLID
      x = norm_layer(x, ntype)
      x = tf.nn.relu(x)
      count_l +=1

    num_down_smp = 2
    for i in range(num_down_smp):
      mult = 2**(i+1)
      with tf.variable_scope('h'+str(count_l), reuse=tf.AUTO_REUSE) as scope:
        x = conv2d(x, ngf*mult, kernel_h=3, stride_h=2, padding='SAME') #[batch, h, w, 128 (256)]
        x = norm_layer(x, ntype)
        x = tf.nn.relu(x)
      count_l +=1

    for i in range(num_block):
      is_dropout=True #for selective droput to layers
      x = res_block(x, ngf*mult, name='res_block_'+str(count_l), is_dropout=is_dropout)
      count_l +=1

    '''
    (upsampling) deconv layer
    '''
    num_up_smp = 2
    for i in range(num_up_smp):
      mult = 2**(num_up_smp -i)
      with tf.variable_scope('h'+str(count_l), reuse=tf.AUTO_REUSE) as scope:
        x = deconv2d(x, int(ngf*mult/2), kernel_h=3, stride_h=2, padding='SAME')
        x = norm_layer(x, ntype)
        x = tf.nn.relu(x)
      count_l+=1

    '''
    output layer
    '''

    with tf.variable_scope('h_out', reuse=tf.AUTO_REUSE) as scope:
      # reflection 2d
      x = conv2d(x, 3, kernel_h=7, stride_h=1, padding='SAME') #Valid
      x = tf.nn.tanh(x)
      output = tf.add(x, input)/2.0
      
    return output


def discriminator(input,  ndf=64, num_layer=3, ntype='batch'):
  ndf = ndf
  with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:
    with tf.variable_scope('h0', reuse=tf.AUTO_REUSE):
      x = leak_relu(conv2d(input, ndf, kernel_h=4, stride_h=2, name='conv'))

    nf_mult, nf_mult_prev = 1,1

    #Iterative Add convolutional block: conv-norm-leak_relu
    for n in range(1,num_layer+1):
      nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
      with tf.variable_scope('h'+str(n), reuse=tf.AUTO_REUSE) as scope:
        x = conv_block(x, ndf*nf_mult, k=4, s=2, ntype=ntype)

    nf_mult_prev, nf_mult = nf_mult, min(2**num_layer, 8)
    #nf_mult_prev, nf_mult = nf_mult, min(2**(num_layer+1), 8)
    with tf.variable_scope('h'+str(num_layer+1), reuse=tf.AUTO_REUSE) as scope:
      x = conv_block(x, ndf*nf_mult, k=4, s=1, ntype=ntype)

    #build output layer
    with tf.variable_scope('h_out', reuse=tf.AUTO_REUSE) as scope:
      x = conv2d(x, ndf*nf_mult, kernel_h=4, stride_h=1, name='conv')
      x = tf.contrib.layers.flatten(x)
      x = fc_layer(x, 1024, activation='tanh', name='fc1')
      x = fc_layer(x, 1, activation='linear', name='output')
    return x

def get_x_hat(gen_img, real_img, batch_size=1, name='xhat'):
  with tf.name_scope(name=name) as scope:
    ep = tf.random_uniform(shape=[batch_size, 1,1,1], minval=0.0, maxval=1.0)
    result = real_img + ep * (gen_img - real_img)
  return result

if __name__ == '__main__':
  test_input = np.ones([1,256,256,3], dtype=np.float32)
  test_input = tf.constant(test_input, dtype=tf.float32)

  disc = discriminator(test_input)
  #gent = generator(test_input)

  tf_vars = [var for var in tf.trainable_variables() if 'disc' in var.name]
  for idx, var in enumerate(tf_vars):
    print(var)
  print(disc)


  gent = generator(test_input)
  tf_vars = [var for var in tf.trainable_variables() if 'gene' in var.name]
  for idx, var in enumerate(tf_vars):
    print(var)
  print(gent)






