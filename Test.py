#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import tensorflow as tf
import cv2
import os
import sys
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import Supervised_HomographyModel, Unsupervised_HomographyModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
# from StringIO import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *


# Don't generate pyc codes
sys.dont_write_bytecode = True
patch_size = 128
# generate random patch (square)
r = 32

def SetupAll(BasePath):
    """
    Inputs:
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """
    # Image Input Shape
    ImageSize = [patch_size, patch_size, 2]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.jpg'))
    SkipFactor = 1
    for count in range(1, NumImages+1, SkipFactor):
        DataPath.append(BasePath + str(count) + '.jpg')

    return ImageSize, DataPath


def ReadImages(BasePath):
    """
    Inputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    # Generate random image
    RandIdx = random.randint(1, 1000)

    RandImageName = BasePath + str(RandIdx) + '.jpg'
    I1 = cv2.imread(RandImageName)

    if(I1 is None):
        # OpenCV returns empty list if image is not read!
        print('ERROR: Image I1 cannot be read')
        sys.exit()

    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    img = np.float32(I1)
    img = cv2.resize(img, (240, 320))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # I1 = (I1 - np.mean(I1))/255

    img_gray = (img_gray-np.mean(img_gray))/255
    rand_x = np.random.randint(r, img.shape[1]-r-patch_size)
    rand_y = np.random.randint(r, img.shape[0]-r-patch_size)
    
    point1 = [rand_x, rand_y]
    point2 = [rand_x + patch_size, rand_y]
    point3 = [rand_x + patch_size, rand_y + patch_size]
    point4 = [rand_x, rand_y + patch_size]
    src = [point1, point2, point3, point4]
    src = np.array(src)
    # perform random perturbation in the range [-phi, phi] to the corner points of the patch
    theta = 2 * np.pi * random.random()
    dst = []
    for i in range(len(src)):
        rand_pertub = [src[i][0] + r *
                        np.cos(theta), src[i][1]+r*np.sin(theta)]
        dst.append(rand_pertub)

    H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    H_inv = np.linalg.inv(H)
    img_warp = cv2.warpPerspective(img_gray, H_inv, (img_gray.shape[1], img_gray.shape[0]))
    patch_1 = img_gray[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
    patch_2 = img_warp[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
    patch_stack = np.dstack((patch_1, patch_2))
    H4pt = np.subtract(dst, src)
    src = np.float32(src)
    patch_1 = np.float32(patch_1.reshape(patch_size, patch_size, 1))
    patch_2 = np.float32(patch_2.reshape(patch_size, patch_size, 1))
    return patch_stack, H4pt, src, dst, I1, patch_1, patch_2
                

def TestSupervised(ImgPH, ImageSize, ModelPath, DataPath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    
    patch_stack, H4pt, src, dst, img_orig, _, _ = ReadImages(DataPath)

    # Predict output with forward pass, MiniBatchSize for Test is 1
    H4pt_pred = Supervised_HomographyModel(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()

    
    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        
        Img=np.array(patch_stack).reshape(1, patch_size, patch_size,2)
        FeedDict = {ImgPH: Img}
        Predicted = sess.run(H4pt_pred,FeedDict)

    src_new=src+Predicted.reshape(4,2)
    H4pt_new=dst-src_new
    
    cv2.polylines(img_orig,np.int32([src]),True,(0,255,0), 3)
    cv2.polylines(img_orig,np.int32([dst]),True,(255,0,0), 5)
    cv2.polylines(img_orig,np.int32([src_new]),True,(0,0,255), 5)
    plt.figure()
    plt.imshow(img_orig)
    plt.show()


def TestUnsupervised(ImgPH, ImageSize, ModelPath, DataPath):
    
    CornerPH = tf.placeholder(tf.float32, shape=(1,4,2))
    patch1PH = tf.placeholder(tf.float32, shape=(1, patch_size, patch_size,1))
    patch2PH = tf.placeholder(tf.float32, shape=(1, patch_size, patch_size,1))

    # Predict output with forward pass, MiniBatchSize for Test is 1
    pred_I2, I2, pred_H4pt = Unsupervised_HomographyModel(ImgPH, CornerPH, patch1PH, patch2PH, ImageSize, 1)
    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        patch_stack, H4pt, src, dst, img_orig, patch1, patch2 = ReadImages(DataPath)
        Img=np.array(patch_stack).reshape(1,patch_size, patch_size,2)
        corner =np.array(src).reshape(1, 4, 2)
        patch1 = np.array(patch1).reshape(1, patch_size, patch_size, 1)
        patch2 = np.array(patch2).reshape(1, patch_size, patch_size, 1)
        FeedDict = {ImgPH: Img, CornerPH: corner, patch1PH: patch1, patch2PH: patch2}
        Predicted = sess.run(pred_H4pt, FeedDict)

    src_new=src+Predicted.reshape(4,2)
    
    cv2.polylines(img_orig,np.int32([src]),True,(0,255,0), 3)
    cv2.polylines(img_orig,np.int32([dst]),True,(255,0,0), 5)
    cv2.polylines(img_orig,np.int32([src_new]),True,(0,0,255), 5)
    plt.figure()
    plt.imshow(img_orig)
    plt.show()

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/bernard/CMSC733/proj_1/Phase2/Checkpoints/', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/bernard/CMSC733/proj_1/Phase2/Data/Val/', help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--ModelType', default='Unsup',
                        help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Args = Parser.parse_args()
    ModelType = Args.ModelType
    ModelPath = Args.ModelPath + str(ModelType)+ '/19model.ckpt'
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath

    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, patch_size, patch_size, 2))

    # LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels
    if ModelType == 'Sup':
        TestSupervised(ImgPH, ImageSize, ModelPath, BasePath)
    elif ModelType == 'Unsup':    
        TestUnsupervised(ImgPH, ImageSize, ModelPath, BasePath)
    else:
        print("ERROR: Unknown ModelType !!!")
        sys.exit()
    # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
 
