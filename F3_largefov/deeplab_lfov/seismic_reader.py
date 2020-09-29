#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:10:59 2019

@author: huguang
"""

import os
import random
import tensorflow as tf

def read_labeled_seismic_images_list(seismic_dir):
    seismics = []
    labels_train = []
    #labels_test = []
    
    train_list = os.listdir(seismic_dir)
    for s in train_list:
        seismic_new = (seismic_dir + s)
        seismics.append(seismic_new)
    #从seismics_train随机选取400个作为训练列表
    seismics_train = random.sample(seismics, 470)
    
    for l in seismics_train:
        l = l.replace('/seismic/','/label/')
        labels_train.append(l)
    """
    for m in seismic_test:
        m = m.replace('/seismic/','/label/')
        m = m.replace('.jpeg','.png')
        labels_test.append(m)
    """
    return seismics_train,labels_train
    #return seismics_train,labels_train,seismic_test,labels_test#得到训练数据和标签数据文件路径列表

def read_seismic_images_from_disk(input_queue):
    """Read one image and its corresponding mask with optional pre-processing.
    Args:
      input_queue: tf queue with paths to the image and its mask.
    Returns:
      Two tensors: the decoded image and its mask.
    """
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    img = tf.image.decode_png(img_contents, channels=1)
    label = tf.image.decode_png(label_contents, channels=1)
    
    
    img = tf.image.resize_images(img, [100,100])
    #img = img /255
    
    label = tf.image.resize_images(label, [100,100])
    label = label / 255
    

    return img, label

class seismic_ImageReader(object):
    def __init__(self, seismic_dir, coord):
        self.seismic_dir = seismic_dir
        #self.data_list = data_list
        self.coord = coord
        self.seismic_train_image_list, self.label_train_list= read_labeled_seismic_images_list(self.seismic_dir)
        self.seismic_train_image_list = tf.convert_to_tensor(self.seismic_train_image_list, dtype=tf.string)
        self.label_train_list = tf.convert_to_tensor(self.label_train_list, dtype=tf.string)
        #self.seismic_test_image_list = tf.convert_to_tensor(self.seismic_train_image_list, dtype=tf.string)
        #self.label_test_list = tf.convert_to_tensor(self.label_train_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.seismic_train_image_list, self.label_train_list],shuffle=True)
        self.seismic_train_image, self.seismic_train_label = read_seismic_images_from_disk(self.queue)
        
    def dequeue(self, num_elements):
        image_batch, label_batch = tf.train.batch([self.seismic_train_image, self.seismic_train_label],
                                                  num_elements)
        return image_batch, label_batch
