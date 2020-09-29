#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:37:25 2019

@author: huguang
"""

import tensorflow as tf
from six.moves import cPickle


#打开net_skeleton.ckpt，并读取。
with open("E:/huhuang/deep_lab/F3_0618_1aspp/net_skeleton_simp_aspp.ckpt", "rb") as f:
    net_skeleton = cPickle.load(f)

# The DeepLab-LargeFOV model can be represented as follows:
## input -> [conv-relu](dilation=1, channels=64) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=128) x 2 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=256) x 3 -> [max_pool](stride=2)
##       -> [conv-relu](dilation=1, channels=512) x 3 -> [max_pool](stride=1)
##       -> [conv-relu](dilation=2, channels=512) x 3 -> [max_pool](stride=1) -> [avg_pool？](stride=1)
##       -> [conv-relu](dilation=2？, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=1024) -> [dropout]
##       -> [conv-relu](dilation=1, channels=2) -> [pixel-wise softmax loss]

num_layers = [1, 1, 3, 1, 1]
dilations = [[1],
             [1], 
             [1, 3, 5]]
n_classes = 2
ks = 2

def create_variable(name, shape):
    """利用Xavier初始化变量；
       初始化w变量。
    """
    initialiser = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    #“Xavier”初始化方法是一种很有效的神经网络初始化方法。
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

def create_bias_variable(name, shape):
    """初始化b变量为0
    """
    initialiser = tf.constant_initializer(value=0.0, dtype=tf.float32)
    variable = tf.Variable(initialiser(shape=shape), name=name)
    return variable

class HDCASPPModel(object):
    """
    """
    def __init__ (self, weights_path=None):
        self.variables = self._create_variables(weights_path)
        
    def _create_variables(self, weights_path):
        var = list()
        index = 0
        if weights_path is not None:
            with open(weights_path, "rb") as f:
                weights = cPickle.load(f)
                for name, shape in net_skeleton:
                    var.append(tf.Variable(weights[name],name=name))
                del weights
            
        else:
            for name, shape in net_skeleton:
                if "/w" in name:
                    w = create_variable(name, list(shape))
                    var.append(w)
                else:
                    b = create_bias_variable(name, list(shape))
                    var.append(b)
        return var
    
    def _create_network(self, input_batch, keep_prob):
        current = input_batch
        v_idx = 0
        for b_idx in range(len(dilations) ):
            for l_idx, dilation in enumerate(dilations[b_idx]):
                w = self.variables[v_idx * 2]
                b = self.variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation, padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                v_idx += 1
            
            #移除POOL层
            if b_idx < 1:
                current = tf.nn.max_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
            elif b_idx == 1:
                
                current = tf.nn.max_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
            elif b_idx == 2:
                current = tf.nn.max_pool(current, 
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
        """
                current = tf.nn.avg_pool(current,
                                         ksize=[1, ks, ks, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME')
                
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)
        """
        #ASPP
        ####2
        w = self.variables[v_idx * 2]
        b = self.variables[v_idx * 2 + 1]
        conv_2_fc6 = tf.nn.atrous_conv2d(current, w, 6, padding='SAME')
        conv_2_fc6 = conv_2_fc6 + b
        conv_2_fc6 = tf.nn.relu(conv_2_fc6)
        conv_2_fc6 = tf.nn.dropout(conv_2_fc6, keep_prob=keep_prob)
        
        w = self.variables[v_idx * 2 + 2]
        b = self.variables[v_idx * 2 + 3]
        conv_2_fc7 = tf.nn.conv2d(conv_2_fc6, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_2_fc7 = conv_2_fc7 + b
        conv_2_fc7 = tf.nn.relu(conv_2_fc7)
        conv_2_fc7 = tf.nn.dropout(conv_2_fc7, keep_prob=keep_prob)
        
        w = self.variables[v_idx * 2 + 4]
        b = self.variables[v_idx * 2 + 5]
        conv_2_fc8 = tf.nn.conv2d(conv_2_fc7, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_2_fc8 = conv_2_fc8 + b
        
        ####4
        w = self.variables[v_idx * 2 + 6]
        b = self.variables[v_idx * 2 + 7]
        conv_4_fc6 = tf.nn.atrous_conv2d(current, w, 12, padding='SAME')
        conv_4_fc6 = conv_4_fc6 + b
        conv_4_fc6 = tf.nn.relu(conv_4_fc6)
        conv_4_fc6 = tf.nn.dropout(conv_4_fc6, keep_prob=keep_prob)
        
        w = self.variables[v_idx * 2 + 8]
        b = self.variables[v_idx * 2 + 9]
        conv_4_fc7 = tf.nn.conv2d(conv_4_fc6, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_4_fc7 = conv_4_fc7 + b
        conv_4_fc7 = tf.nn.relu(conv_4_fc7)
        conv_4_fc7 = tf.nn.dropout(conv_4_fc7, keep_prob=keep_prob)
        
        w = self.variables[v_idx * 2 + 10]
        b = self.variables[v_idx * 2 + 11]
        conv_4_fc8 = tf.nn.conv2d(conv_4_fc6, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_4_fc8 = conv_4_fc8 + b

        ####8
        w = self.variables[v_idx * 2 + 12]
        b = self.variables[v_idx * 2 + 13]
        conv_8_fc6 = tf.nn.atrous_conv2d(current, w, 18, padding='SAME')
        conv_8_fc6 = conv_8_fc6 + b
        conv_8_fc6 = tf.nn.relu(conv_8_fc6)
        conv_8_fc6 = tf.nn.dropout(conv_8_fc6, keep_prob=keep_prob)
        
        w = self.variables[v_idx * 2 + 14]
        b = self.variables[v_idx * 2 + 15]
        conv_8_fc7 = tf.nn.conv2d(conv_8_fc6, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_8_fc7 = conv_8_fc7 + b
        conv_8_fc7 = tf.nn.relu(conv_8_fc7)
        conv_8_fc7 = tf.nn.dropout(conv_8_fc7, keep_prob=keep_prob)
        
        w = self.variables[v_idx * 2 + 16]
        b = self.variables[v_idx * 2 + 17]
        conv_8_fc8 = tf.nn.conv2d(conv_8_fc7, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_8_fc8 = conv_8_fc8 + b
        
        ####16
        w = self.variables[v_idx * 2 + 18]
        b = self.variables[v_idx * 2 + 19]
        conv_12_fc6 = tf.nn.atrous_conv2d(current, w, 24, padding='SAME')
        conv_12_fc6 = conv_12_fc6 + b
        conv_12_fc6 = tf.nn.relu(conv_12_fc6)
        conv_12_fc6 = tf.nn.dropout(conv_12_fc6, keep_prob=keep_prob)
        
        w = self.variables[v_idx * 2+ 20]
        b = self.variables[v_idx * 2 + 21]
        conv_12_fc7 = tf.nn.conv2d(conv_8_fc6, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_12_fc7 = conv_12_fc7 + b
        conv_12_fc7 = tf.nn.relu(conv_12_fc7)
        conv_12_fc7 = tf.nn.dropout(conv_12_fc7, keep_prob=keep_prob)
        
        w = self.variables[v_idx * 2 + 22]
        b = self.variables[v_idx * 2 + 23]
        conv_12_fc8 = tf.nn.conv2d(conv_12_fc7, w, strides=[1, 1, 1, 1], padding='SAME')
        conv_12_fc8 = conv_12_fc8 + b
        
        current = conv_2_fc8 + conv_4_fc8 + conv_8_fc8 + conv_12_fc8
        
        return current
    #需看看是否有问题！！！！ 需测试！！！
    def prepare_label(self, input_batch, new_size):
        """Resize masks and perform one-hot encoding.
        Args:
          input_batch: input tensor of shape [batch_size H W 1].
          new_size: a tensor with new height and width.
        Returns:
          Outputs a tensor of shape [batch_size h w 2]
          with last dimension comprised of 0's and 1's only.
        """
        with tf.name_scope('label_encode'):
            # As labels are integer numbers, need to use NN interp.
            input_batch = tf.image.resize_nearest_neighbor(
                input_batch, new_size)
            input_batch = tf.cast(input_batch,dtype=tf.int32)
            #标签转成[batch_size H W 1]
            #input_batch = tf.squeeze(input_batch, squeeze_dims=[3])
            #[batch_size h w ]
            input_batch = tf.one_hot(input_batch, depth=n_classes)
        return input_batch

    def preds(self, input_batch):
        
        """Create the network and run inference on the input batch.
        Args:
          input_batch: batch of pre-processed images.
        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        raw_output = self._create_network(
            tf.cast(input_batch, tf.float32), keep_prob=tf.constant(1.0))
        
        raw_output = tf.image.resize_bilinear(
            raw_output, [100,100])
        #raw_output = tf.image.resize_bilinear(
            #raw_output, tf.shape(input_batch)[1:3, ])
        raw_output1 = tf.nn.softmax(tf.cast(raw_output, tf.float32))
        raw_output = tf.argmax(raw_output, dimension=3)
        raw_output = tf.expand_dims(raw_output, dim=3)  # Create 4D-tensor.
        return raw_output1,tf.cast(raw_output, tf.uint8)
    #需要弄明白！！！！ 需测试！！！
    
    def loss(self, img_batch, label_batch):
        
        """Create the network, run inference on the input batch and compute loss.
        Args:
          input_batch: batch of pre-processed images.
        Returns:
          Pixel-wise softmax loss.
        """
        raw_output = self._create_network(
            tf.cast(img_batch, tf.float32), keep_prob=tf.constant(0.5))
        #raw_output = tf.image.resize_bilinear(raw_output, [400, 300])
        #softmax 损失函数
        #prediction = tf.reshape(raw_output, [-1, n_classes])
        #sigmoid 损失函数
        prediction = tf.reshape(raw_output, [-1, n_classes])
        #prediction = tf.nn.softmax(prediction)

        # Need to resize labels and convert using one-hot encoding.
        label_batch = self.prepare_label(label_batch,tf.stack(raw_output.get_shape()[1:3]))
        #softmax 损失函数
        #gt = tf.reshape(label_batch, [-1, n_classes])
    
        gt = tf.reshape(label_batch, [-1, n_classes])
        weights = tf.constant([0.05,0.95])
        weights = tf.reduce_sum(tf.multiply(gt, weights), -1)

        # Pixel-wise softmax loss.
        loss = tf.losses.softmax_cross_entropy( onehot_labels = gt,logits = prediction,weights = weights)
        
        

        return loss