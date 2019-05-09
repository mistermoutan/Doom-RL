import numpy as np
import tensorflow as tf
import scipy
import scipy.signal
import random
import scipy.misc
import csv
import tensorflow.contrib.slim as slim
import os

import cv2

from vizdoom import *
from utils.network_params import *

def normalized_columns_initializer(std=1.0):
    """
    Description
    ---------------
    Tensorflow zero-mean, std weights initializer.
    
    Parameters
    ---------------
    std  : float, std for the normal distribution
    
    Returns
    ---------------
    _initializer : Tensorflow initializer
    """
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def update_target_graph(from_scope,to_scope):
    """
    Description
    ---------------
    Copies set of variables from one network to the other.
    
    Parameters
    ---------------
    from_scope : String, scope of the origin network
    to_scope   : String, scope of the target network
    
    Returns
    ---------------
    op_holder  : list, variables copied.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(frame, crop, resize):
    """
    Description
    ---------------
    Crop and resize Doom screen frame.
    
    Parameters
    ---------------
    frame  : np.array, screen image
    crop   : tuple, top, bottom, left and right crops
    resize : tuple, new width and height
    
    Returns
    ---------------
    s      : np.array, screen image cropped and resized.
    """
    y2, y1, x1, x2 = crop
    s = frame[y2:y1,x1:x2]
    s = scipy.misc.imresize(s,list(resize))
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s
def process_frame_as_pc(frame, crop, resize):
    """
    Description
    ---------------
    Crop and resize Doom screen frame.
    
    Parameters
    ---------------
    frame  : np.array, screen image
    crop   : tuple, top, bottom, left and right crops
    resize : tuple, new width and height
    
    Returns
    ---------------
    s      : np.array, screen image cropped and resized.
    """
    y2, y1, x1, x2 = crop
    s = frame[y2:y1,x1:x2]
    s = scipy.misc.imresize(s,list(resize))
    #s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

def cropping(s):
    frame = np.asarray(s)
    frame = cv2.resize(frame, (64,64))
    return frame

def discount(x, gamma):
    """
    Description
    ---------------
    Returns gamma-discounted cumulated values of x
    [x0 + gamma*x1 + gamma^2*x2 + ..., 
     x1 + gamma*x2 + gamma^2*x3 + ...,
     x2 + gamma*x3 + gamma^2*x4 + ...,
     ...,
     xN]
    
    Parameters
    ---------------
    x      : list, list of values
    gamma  : float, top, bottom, left and right crops
    
    Returns
    ---------------
    np.array, gamma-discounted cumulated values of x
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
