#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:59:29 2020

@author: huguang
reference:https://blog.csdn.net/u012370185/article/details/94409933
"""
import os
import cv2
import numpy as np
 
class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
 
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
 
    def _fast_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist
    # 输入：预测值和真实值
    # 语义分割的任务是为每个像素点分配一个label
    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
            
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou) 
        
        # -----------------其他指标------------------------------
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum() #总体精确度计算
        #acc_cls = np.diag(hist) / hist.sum(axis=1)#求得每一类像素精确度的平均值？？？
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))#求得总体平均精确度
 
        freq = self.hist.sum(axis=1) / self.hist.sum()#求得标签中各个类别所占的比例
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()#按权值计算交并比
 
        return acc, acc_cls, iou, miou, fwavacc
    

def read_labeled_seismic_images_list(mask_dir,pred_dir):
    seismics = []
    labels = []
    s_l_list = []
    seismic_list = os.listdir(mask_dir)
    for s in seismic_list:
        seismic_new = (mask_dir + s)
        seismics.append(seismic_new)
        
    labels_list = os.listdir(pred_dir)
    for s in labels_list:
        labels_new = (pred_dir + s)
        labels.append(labels_new)
    
    n = len(seismics)
    for i in range(n):
        s_l_list.append([seismics[i],labels[i]])
    #print(seismics)
    #print(s_l_list)
    return s_l_list#得到训练数据和标签数据文件路径列表

if __name__ == '__main__':
    #路径
    """
    label_path = '/Volumes/G/clyde_result_analysis/label_&_pred_test/annotations1/'
    predict_path = '/Volumes/G/clyde_result_analysis/label_&_pred_test/output1/'
    """
    label_path = 'E:/huhuang/seismic_train_data/F3/F3_image_data_100x400/label/'
    predict_path = 'E:/huhuang/seismic_train_data/F3/F3_train_data_1/output_12/merge/'
    
    l_new = read_labeled_seismic_images_list(predict_path,label_path)
    labels = []
    predicts = []
    for im in l_new:
        lab_path = im[1]
        print(lab_path)
        pre_path = im[0]
        print(pre_path)
        label = cv2.imread(lab_path,0)
        
        label = label/255
        label = label.astype(int)
        
        pre = cv2.imread(pre_path,0)
        pre = pre/255
        pre = pre.astype(int)
        labels.append(label)
        predicts.append(pre)

    el = IOUMetric(2)      #类别个数
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
    print('acc: ',acc)
    print('acc_cls: ',acc_cls)
    print('iou: ',iou)
    print('miou: ',miou)
    print('fwavacc: ',fwavacc)
