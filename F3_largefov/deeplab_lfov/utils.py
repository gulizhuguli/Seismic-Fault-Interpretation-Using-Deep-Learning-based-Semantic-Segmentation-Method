#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:23:34 2019

@author: huguang
"""

import os
import tensorflow as tf



def load(saver, sess, ckpt_path):
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restored model parameters from {}".format(ckpt_path))
        
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
