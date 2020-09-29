#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:07:07 2020

@author: huguang
"""
from PIL import Image
import os

def read_labeled_seismic_images_list(seismic_dir):
    
    seismics = []
    s_l_list = []

    seismic_list = os.listdir(seismic_dir)
    for s in seismic_list:
        seismic_new = (seismic_dir + s)
        seismics.append(seismic_new)
        
    
    n = len(seismics)
    for i in range(n):
        s_l_list.append(seismics[i])
    #print(seismics)
    return s_l_list#得到训练数据和标签数据文件路径列表

path = "E:/huhuang/deep_lab/F3_0915_1largefov/output/output705/pred/"
s_list = read_labeled_seismic_images_list(path)
#del(s_list[0])
#print(s_list[0])
print(len(s_list))

def list_of_groups(list_info, per_list_len):
    '''
    :param list_info:   列表
    :param per_list_len:  每个小列表的长度
    :return:
    '''
    list_of_group = zip(*(iter(list_info),) *per_list_len) 
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list

s_l_new = list_of_groups(s_list,4)
#print(s_l_new)
#print(s_l_new[0][0])
#print(s_l_new[0][0][-10:-7] + '.jpeg')
#path = "/Users/apple/Desktop/crop_test/F3_merge/"
#print(path + s_l_new[0][0][-10:-7] + '.jpeg')
#print(str(path + s_l_new[0][0][-10:-7] + '.jpeg'))
#print(len(s_l_new))

def image_compose(image_names,image_s,image_r,image_c,save_path):
    if len(image_names) != image_r * image_c:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")
    to_image = Image.new('L', (image_c * image_s, image_r * image_s)) #创建一个新图
    for y in range(1, image_r + 1):
        for x in range(1, image_c + 1):
            from_image = Image.open(image_names[image_c * (y - 1) + x - 1]).resize(
                (image_s, image_s),Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * image_s, (y - 1) * image_s))
    #label
    name = image_names[0][-10:-7] + '.png'
    #seismic
    #name = image_names[0][-10:-7] + '.jpeg'
    save_new = str(save_path + name)
    return to_image.save(save_new)

s_path = "E:/huhuang/deep_lab/F3_0915_1largefov/output/output705/merge/"

for i in s_l_new:
    image_compose(i,100,1,4,s_path)
