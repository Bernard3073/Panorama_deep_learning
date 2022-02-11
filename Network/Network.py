"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True

def HomographyModel(Img, ImageSize, MiniBatchSize):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """
    H4Pt = Supervised(Img)
   
    return H4Pt

def Supervised(x):
    x = tf.layers.conv2d(inputs=x,padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x, name='relu_1')

    x = tf.layers.conv2d(inputs=x, padding='same',filters=64, kernel_size=[3,3], activation=None) 
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x, name='relu_2')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x, name='relu_3')

    x = tf.layers.conv2d(inputs=x, padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x, name='relu_4')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x, name='relu_5')

    x = tf.layers.conv2d(inputs=x, padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x, name='relu_6')

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)
    
    x = tf.layers.conv2d(inputs=x, padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x, name='relu_7')

    x = tf.layers.conv2d(inputs=x, padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x, name='relu_8')


    x = tf.compat.v1.layers.flatten(x)

    #Fully-connected layers
    x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    x = tf.layers.dropout(x,rate=0.5,noise_shape=None,seed=None,training=True,name=None)
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(inputs=x, units=8, activation=None)

    return x
