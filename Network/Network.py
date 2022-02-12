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
from Misc.TFSpatialTransformer import transformer
from Network.utils import *
# Don't generate pyc codes
sys.dont_write_bytecode = True
patch_size = 32
def Supervised_HomographyModel(x, ImageSize, MiniBatchSI):
    x = tf.layers.conv2d(inputs=x,padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(inputs=x, padding='same',filters=64, kernel_size=[3,3], activation=None) 
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(inputs=x, padding='same',filters=64, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)

    x = tf.layers.conv2d(inputs=x, padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(inputs=x, padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.max_pooling2d(inputs=x, pool_size=[2,2], strides=2)
    
    x = tf.layers.conv2d(inputs=x, padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(inputs=x, padding='same',filters=128, kernel_size=[3,3], activation=None)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = tf.compat.v1.layers.flatten(x)

    #Fully-connected layers
    x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    x = tf.layers.dropout(x,rate=0.5,noise_shape=None,seed=None,training=True,name=None)
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(inputs=x, units=8, activation=None)

    return x
 
def TensorDLT(H4pt, C4A , MiniBatchSize):
    
    pts_1_tile = tf.expand_dims(C4A, [2]) # BATCH_SIZE x 8 x 1
    
    # Solve for H using DLT
    pred_h4p_tile = tf.expand_dims(H4pt, [2]) # BATCH_SIZE x 8 x 1
    # 4 points on the second image
    pred_pts_2_tile = tf.add(pred_h4p_tile, pts_1_tile)


    # Auxiliary tensors used to create Ax = b equation
    M1_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M1,tf.float32),[0]),[MiniBatchSize,1,1])
    M2_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M2,tf.float32),[0]),[MiniBatchSize,1,1])
    M3_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M3,tf.float32),[0]),[MiniBatchSize,1,1])
    M4_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M4,tf.float32),[0]),[MiniBatchSize,1,1])
    M5_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M5,tf.float32),[0]),[MiniBatchSize,1,1])
    M6_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M6,tf.float32),[0]),[MiniBatchSize,1,1])
    M71_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M71,tf.float32),[0]),[MiniBatchSize,1,1])
    M72_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M72,tf.float32),[0]),[MiniBatchSize,1,1])
    M8_tile = tf.tile(tf.expand_dims(tf.constant(Aux_M8,tf.float32),[0]),[MiniBatchSize,1,1])
    Mb_tile = tf.tile(tf.expand_dims(tf.constant(Aux_Mb,tf.float32),[0]),[MiniBatchSize,1,1])

    # Form the equations Ax = b to compute H
    # Form A matrix
    A1 = tf.matmul(M1_tile, pts_1_tile) # Column 1
    A2 = tf.matmul(M2_tile, pts_1_tile) # Column 2
    A3 = M3_tile                   # Column 3
    A4 = tf.matmul(M4_tile, pts_1_tile) # Column 4
    A5 = tf.matmul(M5_tile, pts_1_tile) # Column 5
    A6 = M6_tile                   # Column 6
    A7 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M72_tile, pts_1_tile)# Column 7
    A8 = tf.matmul(M71_tile, pred_pts_2_tile) *  tf.matmul(M8_tile, pts_1_tile)# Column 8

    A_mat = tf.transpose(tf.stack([tf.reshape(A1,[-1,8]),tf.reshape(A2,[-1,8]),\
                                   tf.reshape(A3,[-1,8]),tf.reshape(A4,[-1,8]),\
                                   tf.reshape(A5,[-1,8]),tf.reshape(A6,[-1,8]),\
                                   tf.reshape(A7,[-1,8]),tf.reshape(A8,[-1,8])],axis=1), perm=[0,2,1]) # BATCH_SIZE x 8 (A_i) x 8


    print('--Shape of A_mat:', A_mat.get_shape().as_list())
    # Form b matrix
    b_mat = tf.matmul(Mb_tile, pred_pts_2_tile)
    print('--shape of b:', b_mat.get_shape().as_list())

    # Solve the Ax = b
    H_8el = tf.matrix_solve(A_mat , b_mat)  # BATCH_SIZE x 8.
    print('--shape of H_8el', H_8el)


    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = tf.ones([MiniBatchSize, 1, 1])
    H_9el = tf.concat([H_8el,h_ones],1)
    H_flat = tf.reshape(H_9el, [-1,9])
    H_mat = tf.reshape(H_flat,[-1,3,3])   # BATCH_SIZE x 3 x 3

    return H_mat

def Unsupervised_HomographyModel(Img, C4A, I2, ImageSize, MiniBatchSize):

    H4pt = Supervised_HomographyModel(Img)
    C4A_pts = tf.reshape(C4A,[MiniBatchSize,8])
    print(C4A.get_shape())
    H_mat = TensorDLT(H4pt, C4A_pts, MiniBatchSize)
    img_h = ImageSize[1]
    img_w = ImageSize[0]
    # Constants and variables used for spatial transformer
    M = np.array([[img_w/2.0, 0., img_w/2.0],
                  [0., img_h/2.0, img_h/2.0],
                  [0., 0., 1.]]).astype(np.float32)

    M_tensor  = tf.constant(M, tf.float32)
    M_tile   = tf.tile(tf.expand_dims(M_tensor, [0]), [MiniBatchSize, 1,1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv   = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [MiniBatchSize,1,1])

    y_t = tf.range(0, MiniBatchSize*img_w*img_h, img_w*img_h)
    z =  tf.tile(tf.expand_dims(y_t,[1]),[1,patch_size*patch_size])
    batch_indices_tensor = tf.reshape(z, [-1]) # Add these value to patch_indices_batch[i] for i in range(num_pairs) # [BATCH_SIZE*WIDTH*HEIGHT]

    # Transform H_mat since we scale image indices in transformer
    H_mat = tf.matmul(tf.matmul(M_tile_inv, H_mat), M_tile)
    # Transform image 1 (large image) to image 2
    out_size = (img_h, img_w)

    I1 = tf.slice(Img,[0,0,0,0],[MiniBatchSize,patch_size,patch_size,1])
    print(I1)
    print(Img)
    warped_images, _ = transformer(I1, H_mat, out_size)
    # print(warped_images.get_shape())
    warped_gray_images = tf.reduce_mean(warped_images, 3)

    pred_I2_flat = warped_gray_images

    pred_I2 = tf.reshape(pred_I2_flat, [MiniBatchSize, patch_size, patch_size, 1])


    return pred_I2,I2
