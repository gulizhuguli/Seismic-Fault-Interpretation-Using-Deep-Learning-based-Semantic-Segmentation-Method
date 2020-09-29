#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:51:21 2020

@author: apple
"""

import os
import shutil
import random

def cut_file(s_old_path,s_new_path,l_old_path,l_new_path):
    
    old_list = []
    s_new_list = []
    l_new_list = []
    
    old_list_file = os.listdir(s_old_path)
    
    print(old_list_file)
    #del(old_list_file[0])
    #print(old_list_file)
    
    for s in old_list_file:
        old_list_single = (s_old_path + s)
        old_list.append(old_list_single)
        
    s_new_list = random.sample(old_list, 10)
    print(s_new_list)
    
    for l in s_new_list:
        l = l.replace('/seismic/','/label/')
        l_new_list.append(l)
    print(l_new_list)
    
    for m in s_new_list:
        shutil.move(m,s_new_path)
        #os.remove(n)
    for n in l_new_list:
        shutil.move(n,l_new_path)
        
    return None

path1 = "E:/huhuang/seismic_train_data/F3/F3_train_data_5/train/seismic/"
path2 = "E:/huhuang/seismic_train_data/F3/F3_train_data_5/test/seismic/"
path3 = "E:/huhuang/seismic_train_data/F3/F3_train_data_5/train/label/"
path4 = "E:/huhuang/seismic_train_data/F3/F3_train_data_5/test/label/"
cut_file(path1,path2,path3,path4)