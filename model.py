import tensorflow as tf
from tensorlayer.layers import *

w_init = tf.variance_scaling_initializer(scale=2/(1.04)) # msra initialization
a_init = tf.constant_initializer(0.2) # for PReLU

# VGG for perceptual loss
def vgg19(rgb, reuse=False):
    VGG_MEAN = [103.939, 116.779, 123.68]
    r, g, b = tf.split(rgb*255, 3, 3)
    bgr = tf.concat([b-VGG_MEAN[0], g-VGG_MEAN[1], r-VGG_MEAN[2]], axis=-1)
    with tf.variable_scope('vgg', reuse=reuse):
        input = InputLayer(bgr, name='input')
        net = Conv2d(input, 64, act=tf.nn.relu, name='conv1_1')
        net = Conv2d(net, 64, act=tf.nn.relu, name='conv1_2')
        net = MaxPool2d(net, (2, 2), name='pool1')
        
        net = Conv2d(net, 128, act=tf.nn.relu, name='conv2_1')
        net = Conv2d(net, 128, act=tf.nn.relu, name='conv2_2')
        net = MaxPool2d(net, (2, 2), name='pool2')
        
        net = Conv2d(net, 256, act=tf.nn.relu, name='conv3_1')
        net = Conv2d(net, 256, act=tf.nn.relu, name='conv3_2')
        net = Conv2d(net, 256, act=tf.nn.relu, name='conv3_3')
        net = Conv2d(net, 256, act=tf.nn.relu, name='conv3_4')
        net = MaxPool2d(net, (2, 2), name='pool3')
        
        net = Conv2d(net, 512, act=tf.nn.relu, name='conv4_1')
        net = Conv2d(net, 512, act=tf.nn.relu, name='conv4_2')
        net = Conv2d(net, 512, act=tf.nn.relu, name='conv4_3')
        net = Conv2d(net, 512, act=tf.nn.relu, name='conv4_4')
        net = MaxPool2d(net, (2, 2), name='pool4')
    
        net = Conv2d(net, 512, act=tf.nn.relu, name='conv5_1')
        net = Conv2d(net, 512, act=tf.nn.relu, name='conv5_2')
        net = Conv2d(net, 512, act=tf.nn.relu, name='conv5_3')
        net = Conv2d(net, 512, act=tf.nn.relu, name='conv5_4')
    return net.outputs

# RDNet
def rdnet(img_v, img, hei, wid, reuse=False):
    input = tf.concat([img_v, img], -1)
    with tf.variable_scope('retinex', reuse=reuse):
        n = InputLayer(input, name='in')
        n11 = conv_k3(n, 32, name='l1/cv1')
        n12 = conv_k3(n11, 32, name='l1/cv2')
        
        n21 = conv_k3(n12, 64, s=2, name='l2/cv1')
        n22 = conv_k3(n21, 64, name='l2/cv2')
        
        n31 = conv_k3(n22, 128, s=2, name='l3/cv1')
        n32 = conv_k3(n31, 128, name='l3/cv2')
        n33 = conv_k3(n32, 128, name='l3/cv3')
        n34 = conv_k3(n33, 128, name='l3/cv4')
        
        n23 = upsample_and_concat(n34, n22, 64, hei//2, wid//2, name='l2/cv3')
        n24 = conv_k3(n23, 64, name='l2/cv4')
        
        n24.outputs = tf.image.resize_nearest_neighbor(n24.outputs, [hei, wid])
        n13 = conv_k3(n24, 32, name='l1/cv3')
        n14 = conv_k3(n13, 32, name='l1/cv4')
        n14.outputs = tf.concat([n14.outputs, n13.outputs, n12.outputs, n11.outputs], -1)
        n15 = conv_k1(n14, 64, name='l1/cv5')
        
        n = conv_k3(n15, 4, act='none', name='out')
        img_i = tf.expand_dims(tf.sigmoid(n.outputs[:, :, :, 0]), -1) # 1 channel
        img_r = tf.sigmoid(n.outputs[:, :, :, 1:4]) # 3 channels
    return img_i, img_r

# FENet
def fenet(crm, img, img_r, hei, wid, reuse=False):
    input = tf.concat([crm, img, img_r], -1)    
    with tf.variable_scope('fusion', reuse=reuse):
        n = InputLayer(input, name='in')
        n11 = conv_k3(n, 32, name='l1/cv1')
        n12 = conv_k3(n11, 32, name='l1/cv2')
        
        n21 = conv_k3(n12, 64, s=2, name='l2/cv1')
        n22 = conv_k3(n21, 64, name='l2/cv2')
        
        n31 = conv_k3(n22, 128, s=2, name='l3/cv1')
        n32 = conv_k3(n31, 128, name='l3/cv2')
        n33 = conv_k3(n32, 128, name='l3/cv3')
        n34 = conv_k3(n33, 128, name='l3/cv4')
        
        n23 = upsample_and_concat(n34, n22, 64, hei//2, wid//2, name='l2/cv3')
        n24 = conv_k3(n23, 128, name='l2/cv4')
        n25 = conv_k3(n24, 64, name='l2/cv5')
        
        n13 = upsample_and_concat(n25, n12, 32, hei, wid, name='l1/cv3')
        n14 = conv_k3(n13, 64, name='l1/cv4')
        n15 = conv_k3(n14, 64, name='l1/cv5')
        n = conv_k3(n15, 3, act='none', name='out')
    return n.outputs

# 3x3conv + padding
def conv_k3(input, c, s=1, act='prelu', name=''):
    temp = input.outputs
    if s == 1:
        input.outputs = tf.pad(temp, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
        n = Conv2d(input, c, padding='valid', W_init=w_init, name=name)
    else:
        assert s == 2 # stride = 1 or 2
        input.outputs = tf.pad(temp, [[0, 0], [1, 0], [1, 0], [0, 0]], 'SYMMETRIC')
        n = Conv2d(input, c, (3, 3), (2, 2), padding='valid', W_init=w_init, name=name)
    input.outputs = temp
    
    if act == 'prelu':
        n = PReluLayer(n, a_init=a_init, name=name+'/pr')
    elif act == 'lrelu':
        n.outputs = tf.nn.leaky_relu(n.outputs)
    elif act == 'relu':
        n.outputs = tf.nn.relu(n.outputs)
    return n

# 1x1conv
def conv_k1(input, c, act='none', name=''):
    n = Conv2d(input, c, (1, 1), W_init=w_init, name=name)
    
    if act == 'prelu':
        a_init = tf.constant_initializer(0.2)
        n = PReluLayer(n, a_init=a_init, name=name+'/pr')
    elif act == 'lrelu':
        n.outputs = tf.nn.leaky_relu(n.outputs)
    elif act == 'relu':
        n.outputs = tf.nn.relu(n.outputs)
    return n

# NN + 3x3conv + concat
def upsample_and_concat(input1, input2, c, hei, wid, act='prelu', name=''):
    input1.outputs = tf.image.resize_nearest_neighbor(input1.outputs, [hei, wid])
    n = conv_k3(input1, c, act=act, name=name)
    n.outputs = tf.concat([n.outputs, input2.outputs], -1)
    return n

# weighted TV loss for I
def wtv_loss(img_v, img_i, size):
    dy = img_i[:, 1:, :, :] - img_i[:, :size - 1, :, :]
    dy = tf.pad(dy, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
    dx = img_i[:, :, 1:, :] - img_i[:, :, :size - 1, :]
    dx = tf.pad(dx, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
    
    vdy = img_v[:, 1:, :, :] - img_v[:, :size - 1, :, :]
    vdy = tf.pad(vdy, [[0, 0], [0, 1], [0, 0], [0, 0]], 'SYMMETRIC')
    vdx = img_v[:, :, 1:, :] - img_v[:, :, :size - 1, :]
    vdx = tf.pad(vdx, [[0, 0], [0, 0], [0, 1], [0, 0]], 'SYMMETRIC')
    vdy = MeanPool(vdy) * 9
    vdx = MeanPool(vdx) * 9
    
    wy = tf.divide(tf.square(dy), tf.abs(vdy) * tf.abs(dy) + 1e-3)
    wx = tf.divide(tf.square(dx), tf.abs(vdx) * tf.abs(dx) + 1e-3)
    return tf.reduce_mean(wy) + tf.reduce_mean(wx)

# close operation for V
def close_op(img_v):
    out = tf.layers.max_pooling2d(img_v, 3, 1, padding='same')
    return -tf.layers.max_pooling2d(-out, 3, 1, padding='same')

# stretch for I
def stretch(img_v, img_i):
    i_min = tf.reduce_min(img_i)
    v_min = tf.reduce_min(img_v)
    i_max = tf.reduce_max(img_i)
    v_max = tf.reduce_max(img_v)
    return tf.divide(img_i - i_min, i_max - i_min) * (v_max - v_min) + v_min

# CRM function
def CRM(img, img_i, a=-0.3293, b=1.1258):
    k = tf.minimum(tf.divide(1, img_i), 7) ** a
    out = (img ** k) * tf.exp((1 - k) * b)
    return tf.clip_by_value(out, 0, 1)
