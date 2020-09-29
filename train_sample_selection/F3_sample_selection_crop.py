#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:09:30 2020

@author: huguang
"""

from PIL import Image
import random
import tensorflow as tf
import os

def corp_random(seismic_path,label_path,h,w,seismic_crop_dir,label_crop_dir):
    
    seismic = Image.open(seismic_path)
    label = Image.open(label_path)
    img_size = seismic.size
    m = img_size[0]    #读取图片的宽度
    n = img_size[1]     #读取图片的高度
    
    box1 = (0, 0, 100, 100) # 设置图像裁剪区域
    box2 = (100, 0, 200, 100) # 设置图像裁剪区域
    box3 = (200, 0, 300, 100) # 设置图像裁剪区域
    box4 = (300, 0, 400, 100) # 设置图像裁剪区域
    #seismic
    image1 = seismic.crop(box1) # 图像裁剪
    image2 = seismic.crop(box2) # 图像裁剪
    image3 = seismic.crop(box3) # 图像裁剪
    image4 = seismic.crop(box4) # 图像裁剪
    image1.save(seismic_crop_dir + seismic_path[-7:-4] + "_01.png") # 存储裁剪得到的图像
    image2.save(seismic_crop_dir + seismic_path[-7:-4] + "_02.png") # 存储裁剪得到的图像
    image3.save(seismic_crop_dir + seismic_path[-7:-4] + "_03.png") # 存储裁剪得到的图像
    image4.save(seismic_crop_dir + seismic_path[-7:-4] + "_04.png") # 存储裁剪得到的图像
    #label
    label1 = label.crop(box1) # 图像裁剪
    label2 = label.crop(box2)
    label3 = label.crop(box3)
    label4 = label.crop(box4)# 图像裁剪
    label1.save(label_crop_dir + label_path[-7:-4] + "_01.png") # 存储裁剪得到的图像
    label2.save(label_crop_dir + label_path[-7:-4] + "_02.png") # 存储裁剪得到的图像
    label3.save(label_crop_dir + label_path[-7:-4] + "_03.png") # 存储裁剪得到的图像
    label4.save(label_crop_dir + label_path[-7:-4] + "_04.png") # 存储裁剪得到的图像
    """
    w = 100                  #设置你要裁剪的小图的宽度
    h = 100                   #设置你要裁剪的小图的高度
    """
    for i in range(20):         #裁剪为10张随机的小图
        x = random.randint(0, m-w)       #裁剪起点的x坐标范围
        y = random.randint(0, n-h)        #裁剪起点的y坐标范围
        s = seismic.crop((x, y, x+w, y+h))     #裁剪区域
        l = label.crop((x, y, x+w, y+h))
        i = str(i + 5)
        n_idex = i.zfill(2)
        s.save(seismic_crop_dir + seismic_path[-7:-4] + "_" + n_idex + ".png")      #str(i)是裁剪后的编号，此处是0到9
        l.save(label_crop_dir + label_path[-7:-4] + "_" + n_idex + ".png")
        
    return None

"""
# single_test
s_path = "/Users/apple/Desktop/crop_test/F3_s/121.png"
l_path ="/Users/apple/Desktop/crop_test/F3_l/121.png"
h = 100
w = 100
s_save = "/Users/apple/Desktop/crop_test/F3_s_save/"
l_save = "/Users/apple/Desktop/crop_test/F3_l_save/"
corp_random(s_path,l_path,h,w,s_save,l_save)
"""

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
    #print(s_l_list)
    return s_l_list#得到训练数据和标签数据文件路径列表

seismic_dir = "E:/huhuang/seismic_train_data/F3/F3_image_data_100x400/seismic/"
label_dir = "E:/huhuang/seismic_train_data/F3/F3_image_data_100x400/label/"
list1 = read_labeled_seismic_images_list(seismic_dir,label_dir)
newlist = random.sample(list1, 20)
print(newlist)


w = 100
h = 100
seismic_crop_dir = "E:/huhuang/seismic_train_data/F3/F3_train_data_5/train/seismic/"
label_crop_dir = "E:/huhuang/seismic_train_data/F3/F3_train_data_5/train/label/"

for m in newlist:
    corp_random(m[0],m[1],h,w,seismic_crop_dir,label_crop_dir)


