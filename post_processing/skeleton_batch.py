#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:42:37 2020

@author: apple
"""
import cv2
from skimage import morphology,color
import os

def skeleton_batch(img,save_patch):
    
    img_new = cv2.imread(img)
    image=color.rgb2gray(img_new)
    #image=1-image #反相
    #实施骨架算法
    skeleton =morphology.skeletonize(image)
    #print(skeleton + 0)
    #skeleton_img = Image.fromarray(skeleton*255, 'L')
    cv2.imwrite(save_patch + img[-7:-4] + "_skt.png", skeleton*255)


def read_images_list(img_dir):
    
    img_list_all = []
    
    img_list = os.listdir(img_dir)
    for s in img_list:
        img_new = (img_dir + s)
        img_list_all.append(img_new)
    print(img_list_all)
    return img_list_all#得到训练数据和标签数据文件路径列表

img_dir = "E:/huhuang/deep_lab/costa0915_1largefov/output/output1170/merge/"
img_list = read_images_list(img_dir)


save_patch = "E:/huhuang/deep_lab/costa0915_1largefov/output/output1170/skt/"

for i in img_list:
    skeleton_batch(i,save_patch)
