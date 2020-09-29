# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:18:48 2019

@author: Administrator
"""
from __future__ import print_function
"""Run DeepLab-LargeFOV on a given image.
This script computes a segmentation mask for a given image.
"""
import argparse
from datetime import datetime
import os
import sys
import time
from PIL import Image
import tensorflow as tf
import numpy as np
from vgg_hdc_aspp import HDCASPPModel
import matplotlib.pyplot as plt
tf.reset_default_graph()
"""
实测可以用
"""
IMG_DIR = "E:/huhuang/s_image_data/F3_result/seismic/024_3.jpeg"

SAVE_DIR = 'E:/huhuang/deep_lab/F3_0608/output1/'

WEIGHTS_PATH = 'E:/huhuang/deep_lab/F3_0608/checkpoints/deeplab_lfov_5k/'
N_CLASSES = 2


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="DeepLabLFOV Network Inference.")
    parser.add_argument("--img_path", type=str,
                        help="Path to the RGB image file.", default=IMG_DIR)
    parser.add_argument("--model_weights", type=str,
                        help="Path to the file with model weights.", default=WEIGHTS_PATH)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()



def main():
    """Create the model and start the inference process."""
    args = get_arguments()
    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(args.img_path), channels=1)
    # Create network.
    net = HDCASPPModel()
    # Which variables to load.
    
    # Predictions.
    prob,pred = net.preds(tf.expand_dims(img, dim=0))

    # Set up TF session and initialize variables.
    trainable = tf.trainable_variables()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    #init = tf.global_variables_initializer()
    
    sess.run(tf.global_variables_initializer())
    
    # Load weights.
    
    ckpt = tf.train.get_checkpoint_state(args.model_weights)
    if ckpt and ckpt.model_checkpoint_path:
        checkpoint_path = os.path.join(args.model_weights, 'model.ckpt-900')
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
    """
    # Perform inference.
    
    pred1,pred2 = sess.run([prob,pred])
    #print(pred1)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    msk1 = np.array(pred1)[0, :, :, 0]
    print(msk1)
    im1 = Image.fromarray(msk1)
    if im1.mode == "F":
        im1 = im1.convert('RGB')
    im1.save(args.save_dir + 'prob024_3.png')
    
    print('The output file has been saved to {}'.format(
        args.save_dir + 'prob024_3.png'))
    """
    msk3 = np.array(pred1)[0, :, :, 1]
    print(msk3)
    im3 = Image.fromarray(msk1*255)
    if im3.mode == "F":
        im3 = im3.convert('RGB')
    im1.save(args.save_dir + 'prob1024_3.png')
    
    print('The output file has been saved to {}'.format(
        args.save_dir + 'prob1024_3.png'))
    #print(pred2)
    """
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    msk2 = np.array(pred2)[0, :, :, 0]
    
    im2 = Image.fromarray(msk2*255)
    """
    if im2.mode == "F":
        im2 = im2.convert('RGB')
    """
    im2.save(args.save_dir + 'mask024_3.png')
    
    print('The output file has been saved to {}'.format(
        args.save_dir + 'mask024_3.png'))
if __name__ == '__main__':
    main()