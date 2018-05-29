import tensorflow as tf

def concat(tensors, axis, *args, **kwargs):
  return tf.concat(tensors, axis, *args, **kwargs)


def adv_loss(logit):
  #loss = sum of (-discriminative loss of blurred image)
  #return loss
  pass

def content_loss(fake_img_feat, real_img_feat):
  #Calculate the feature difference of fake_img & real_img faeture
  #return loss
  #pass


def conv2d(input_, output_dim, kernel_h=3, kernel_w=None, stride_h=1, stride_w=None, padding='VALID', reuse=False, initializer=None, use_bias = True, name="conv2d"):
  
  if kernel_w == None: kernel_w = kernel_h
  if stride_w == None: stride_w = stride_h
  if initializer == None: initializer = tf.contrib.layers.xavier_initializer()

  with tf.variable_scope(name, reuse = tf.AUTO_REUSE)
    if reuse==True: scope.reuse_variables()
    w = tf.get_variable('w', [kernel_h, kernel_w, input_.get_shape()[-1], output_dim],
      initializer=initializer)
    conv = tf.nn.conv2d(input_, w, strides=[1,stride_h, stride_w, 1], padding=padding)

    if use_bias:
      b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
      conv = tf.nn.bias_add(conv, b)

    return conv

def res_block():
  pass

def fc_layer(input_, output_dim, initializer = tf.truncated_normal_initializer(stddev=0.02), activation='linear', reuse=False, name=None):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name or "Linear") as scope:
    if reuse==True: scope.reuse_variables()
    if len(shape) > 2 : input_ = tf.layers.flatten(input_)
    w = tf.get_variable("fc_w", [shape[1], output_dim], dtype=tf.float32, initializer = initializer)
    b = tf.get_variable("fc_b", [output_dim], initializer = tf.constant_initializer(0.0))

    result = tf.matmul(input_, w) + b

    if activation == 'linear':
      return result
    elif activation == 'relu':
      return tf.nn.relu(result)
    elif activation == 'sigmoid':
      return tf.nn.sigmoid(result)
