#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:05:05 2019

@author: huguang
"""

from __future__ import print_function


from vgg_hdc_aspp import HDCASPPModel, seismic_ImageReader
from vgg_hdc_aspp.utils import load, save
import numpy as np
import argparse
import tensorflow as tf
import os
import time
import random
import matplotlib.pyplot as plt
tf.reset_default_graph()

BATCH_SIZE = 10
SEISMIC_SLICES_DIRECTORY =  "E:/huhuang/seismic_train_data/F3/F3_train_data_1/train/seismic/"#训练切片数据所在的文件夹   提供数据
LEARNING_RATE = 1e-4    #学习率
NUM_STEPS =1411    #训练轮数
RANDOM_SCALE = True
SAVE_NUM_IMAGES = 2   #保存总的图片数量？
SAVE_PRED_EVERY = 47  #每500保存一次？
SAVE_DIR = 'E:/huhuang/deep_lab/F3_0618_1aspp/images/'#保存的文件夹
SNAPSHOT_DIR = 'E:/huhuang/deep_lab/F3_0618_1aspp/checkpoints/deeplab_lfov_3k/'
RESTORE_FROM = SNAPSHOT_DIR + 'model.ckpt'
WEIGHTS_PATH = None

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--seismic_dir", type=str, default=SEISMIC_SLICES_DIRECTORY,
                        help="Path to the directory containing the SEISMIC slices dataset.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")#学习率
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")#训练轮数
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")#从哪里恢复模型参数？？
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")#保存预测数据地方
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")#图片被保存的总数
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")#
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")#保存模型快照的路径
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")#保存caffemodel的weights的路径，如果没有则随机初始化
    #最后采用对象的parse_args获取解析的参数，
    return parser.parse_args()



def main():
    
    args = get_arguments()
    coord = tf.train.Coordinator()
    
    with tf.name_scope("create_inputs"):
        reader = seismic_ImageReader(args.seismic_dir,coord)
        seismic_image_batch, label_batch = reader.dequeue(args.batch_size)
        
    net = HDCASPPModel(args.weights_path)
    
    loss = net.loss(seismic_image_batch, label_batch)
    
    optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    
    trainable = tf.trainable_variables()
    optim = optimiser.minimize(loss, var_list=trainable)
    prob,pred = net.preds(seismic_image_batch)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()
    sess.run(init)
    saver = tf.train.Saver(var_list=trainable, max_to_keep=30)
    
    """
    if args.restore_from is not None:
        saver.restore(sess, args.restore_from)
        print("加载模型成功！" + args.restore_from)
    else:
        print("模型加载失败！" + args.restore_from)
    """
    
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for step in range(args.num_steps):
        start_time = time.time()
        
        if step % args.save_pred_every == 0:
            
            loss_value, seismic_images, labels, probs, preds, _ = sess.run([loss, seismic_image_batch, label_batch,prob, pred, optim])
            
            fig, axes = plt.subplots(args.save_num_images, 5, figsize = (20, 20))
            for i in range(args.save_num_images):
                axes.flat[i * 5 ].set_title('seismic')#设置title为'data'
                #images, labels=image_batch, label_batch=reader.dequeue(args.batch_size)
                #=tf.train.batch(16）得到的是images, labels的tensor(16,h,w,3),(16,h,w,1)
                #显示原图
                axes.flat[i * 5 ].imshow(seismic_images[i,:, :, 0],cmap = 'gray')
                #显示mask图
                axes.flat[i * 5 + 1].set_title('label')#设置title为'mask'
                axes.flat[i * 5 + 1].imshow(labels[i,:, :, 0],cmap = 'gray')
                #显示pred的图
                axes.flat[i * 5 + 2].set_title('pred')#设置title为'pred'
                axes.flat[i * 5 + 2].imshow(preds[i,:, :, 0],cmap = 'gray')
                
                axes.flat[i * 5 + 3].set_title('prob_0')#设置title为'prob_0'
                axes.flat[i * 5 + 3].imshow(probs[i,:, :, 0])
                
                axes.flat[i * 5 + 4].set_title('prob_1')#设置title为'prob_1'
                axes.flat[i * 5 + 4].imshow(probs[i,:, :, 1])
                #保存到文件夹
                """
                print(np.amax(probs[i,:, :, 0]))
                print(np.amax(probs[i,:, :, 1]))
                print(probs[i,:, :, 0].shape)
                print(probs[i,:, :, 1].shape)
                print(probs[i,:, :, 0])
                print(probs[i,:, :, 1])
                """
            plt.savefig(args.save_dir + str(start_time) + ".png")
            plt.close(fig)
            
            save(saver, sess, args.snapshot_dir, step)

        else:
            loss_value, _ = sess.run([loss, optim])
        duration = time.time() - start_time
        
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    
    coord.request_stop()
    coord.join(threads)
    
start_time1 = time.time()

if __name__ == '__main__':
    main()

elapsed = (time.time() - start_time1)/3600
print('run time =  {:.3f} h'.format(elapsed))