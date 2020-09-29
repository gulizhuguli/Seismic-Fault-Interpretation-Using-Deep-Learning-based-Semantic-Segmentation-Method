#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:45:33 2020

@author: apple
"""

import cv2
import numpy as np
import os

def fuse(img1,img2,save_path):
    imgs = cv2.imread(img1)
    imgl = cv2.imread(img2)
    merge = cv2.addWeighted(imgs,1,imgl,1,0)
    cv2.imwrite( save_path + img1[-7:-4] + '_merge.png', merge)
    
    return None

def read_labeled_seismic_images_list(seismic_dir,label_dir):
    
    seismics = []
    labels = []
    s_l_list = []

    seismic_list = os.listdir(seismic_dir)
    for s in seismic_list:
        seismic_new = (seismic_dir + s)
        seismics.append(seismic_new)
        

    labels_list = os.listdir(label_dir)
    for s in labels_list:
        labels_new = (label_dir + s)
        labels.append(labels_new)
    
    n = len(seismics)
    for i in range(n):
        s_l_list.append([seismics[i],labels[i]])
    #print(seismics)
    #print(s_l_list[1][1])
    return s_l_list#得到训练数据和标签数据文件路径列表

seismic_dir = "E:/huhuang/seismic_train_data/F3/F3_image_data_100x400/seismic/"
label_dir = "E:/huhuang/deep_lab/F3_0915_1largefov/output/output564/merge/"
list_merge = read_labeled_seismic_images_list(seismic_dir,label_dir)

save_path = "E:/huhuang/deep_lab/F3_0915_1largefov/output/output564/funsion_cnn/"
for i in list_merge:
    
    fuse(i[0],i[1],save_path)