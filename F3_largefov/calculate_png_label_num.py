#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:02:02 2019

@author: huguang
"""

import tensorflow as tf
import numpy as np
import os

label_1_total = []
label_0_total = []

def read_labeled_seismic_images_list(seismic_dir):
    
    labels_path = []
    train_list = os.listdir(seismic_dir)
    for s in train_list:
        labels_new = (seismic_dir + s)
        labels_path.append(labels_new)
        
    print(labels_path)
    return labels_path#得到训练数据和标签数据文件路径列表

labels_path_list=read_labeled_seismic_images_list('E:/huhuang/seismic_train_data/F3/F3_train_data_3/train/label/')


def cal_jpeg_lab_num(label_data):
    
    label_contents = tf.gfile.FastGFile(label_data,'rb').read()

    label =  tf.image.decode_png(label_contents, channels=1)
    #print(label)
    #Clyde
    #label = tf.image.resize_images(label, [300,400])
    #F3
    label = tf.image.resize_images(label, [100,100])
    #label = tf.reshape(label, [400,300])
    #label = tf.reshape(label, [300,400])
    #label = tf.squeeze(label, squeeze_dims=[2])
    #print(label)

    sess = tf.Session()
    
    #print(label.eval(session=sess))
    label = label.eval(session=sess)
    label = label/255
    #print(np.amax(label.eval(session=sess)))
    """
    one = tf.ones_like(label)
    zero = tf.zeros_like(label)
    label = tf.where(label > 0 , x=one, y=zero)
    
    print(label.eval(session=sess))
    print(label)
    #print(np.amax(label))
    label_1 = str(label.eval(session=sess).tolist()).count("1")
    """
    label_1 = str(label.tolist()).count("1.")
    label_0 = str(label.tolist()).count("0.")
    #print(label_1)
    #print(label_0)
    #print(int(str(label_1)) + int(str(label_0)))
    #print(int(str(label_0)) / int(str(label_1)))
    return label_1,label_0
#label_data = "/Volumes/G/F3/F3_image_data_100x400/F3_label_png_100x400/002.png"
#cal_jpeg_lab_num(label_data)


for i in labels_path_list:
    label_1,label_0 = cal_jpeg_lab_num(i)
    label_1_total.append(label_1)
    label_0_total.append(label_0)
    
#print(label_1_total)
#print(label_0_total)
label_1_num = sum(label_1_total)
print(label_1_num)
label_0_num = sum(label_0_total)
print(label_0_num)
label_sum = sum(label_1_total) + sum(label_0_total)
print(label_sum)
rat_1 = label_1_num / label_sum
print(rat_1)
rat_0 = label_0_num / label_sum
print(rat_0)