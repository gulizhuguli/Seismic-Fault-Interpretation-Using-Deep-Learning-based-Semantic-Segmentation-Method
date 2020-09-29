# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:18:48 2019

@author: Guang Hu
"""
from __future__ import print_function
"""
This script computes a segmentation mask for a given image.
"""
import tensorflow as tf
tf.reset_default_graph()
import os
from PIL import Image
import numpy as np
from deeplab_lfov import DeepLabLFOVModel
import cv2
import time
"""
实测可以用
"""

SAVE_DIR = "E:/huhuang/deep_lab/F3_0915_1largefov/output/output705/pred/"
WEIGHTS_PATH = 'E:/huhuang/deep_lab/F3_0915_1largefov/checkpoints/deeplab_lfov_1k/'

def inference(IMG,WEIGHTS_PATH,SAVE_DIR):
    """Create the model and start the inference process."""
    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(IMG), channels=1)
    # Create network.
    net = DeepLabLFOVModel()
    # Which variables to load.
    # Predictions.
    pred,prob = net.preds(tf.expand_dims(img, dim=0))
    # Set up TF session and initialize variables.
    trainable = tf.trainable_variables()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #init = tf.global_variables_initializer()
    sess.run(tf.global_variables_initializer())
    
    # Load weights.
    ckpt = tf.train.get_checkpoint_state(WEIGHTS_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        checkpoint_path = os.path.join(WEIGHTS_PATH, 'model.ckpt-705')
        loader = tf.train.Saver(var_list=trainable)
        loader.restore(sess, checkpoint_path)
        print("加载模型成功！" + checkpoint_path)
    else:
        print("模型加载失败！" + checkpoint_path)
    """
    tvs = [v for v in tf.trainable_variables()]
    for v in tvs:
        print(v.name)
        print(sess.run(v))
    # Perform inference.
    """
    pred1,pred2 = sess.run([pred,prob])
    """
    print(pred1)
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    msk1 = np.array(pred1)[0, :, :, 1]
    im1 = Image.fromarray(msk1*255)
    if im1.mode == "F":
        im1 = im1.convert('RGB')
    im1.save(SAVE_DIR + 'prob'+ IMG[-8 : -5] +'.png')
    print('The output file has been saved to {}'.format(
        SAVE_DIR + 'prob'+ IMG[-8 : -5] +'.png'))
    #print(pred2)
    """
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    msk2 = np.array(pred2)[0, :, :, 0]
    im2 = Image.fromarray(msk2*255)
    im2.save(SAVE_DIR + 'mask'+ IMG[-10 : -4] +'.png')
    
    print('The output file has been saved to {}'.format(
        SAVE_DIR + 'mask'+ IMG[-10 : -4] +'.png'))
    
    
    return im2

def read_labeled_seismic_images_list(seismic_dir):
    
    seismics_list = []
    train_list = os.listdir(seismic_dir)
    for s in train_list:
        seismic_new = (seismic_dir + s)
        seismics_list.append(seismic_new)
        
    #print(seismics)
    #print(s_l_list)
    return seismics_list#得到训练数据和标签数据文件路径列表

l_new=read_labeled_seismic_images_list("E:/huhuang/seismic_train_data/F3/F3_image_data_100x400_crop/seismic/")


start_time = time.time()
for m in l_new:
    inference(m,WEIGHTS_PATH,SAVE_DIR)
    tf.reset_default_graph()
    
    
elapsed = (time.time() - start_time) / 60
print('run time = {:.3e} min'.format(elapsed))
print('Image saving is finished!')