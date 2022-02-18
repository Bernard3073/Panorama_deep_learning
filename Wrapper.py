#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s):
Chahat Deep Singh (chahat@terpmail.umd.edu)
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

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
    img_warp = cv2.warpPerspective(
        img_gray, H_inv, (img_gray.shape[1], img_gray.shape[0]))
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

    Img = np.array(patch_stack).reshape(1, patch_size, patch_size, 2)
    # Predict output with forward pass, MiniBatchSize for Test is 1
    H4pt_pred = Supervised_HomographyModel(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum(
            [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        FeedDict = {ImgPH: Img}
        res = sess.run(H4pt_pred, FeedDict)

    dst_pred = src + res.reshape(4,2)

    cv2.polylines(img_orig,np.int32([src]),True,(255, 0, 0), 3)
    # cv2.polylines(img_orig,np.int32([dst]),True,(0, 255, 0), 3)
    cv2.polylines(img_orig,np.int32([dst_pred]),True,(0,0,255), 5)
    plt.figure()
    plt.imshow(img_orig)
    plt.show()
    H = cv2.getPerspectiveTransform(src, dst_pred)
    return H


def TestUnsupervised(ImgPH, ImageSize, ModelPath, DataPath):

    CornerPH = tf.placeholder(tf.float32, shape=(1, 4, 2))
    patch1PH = tf.placeholder(tf.float32, shape=(1, patch_size, patch_size, 1))
    patch2PH = tf.placeholder(tf.float32, shape=(1, patch_size, patch_size, 1))

    # Predict output with forward pass, MiniBatchSize for Test is 1
    pred_I2, I2, pred_H4pt = Unsupervised_HomographyModel(
        ImgPH, CornerPH, patch1PH, patch2PH, ImageSize, 1)
    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum(
            [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        patch_stack, H4pt, src, dst, img_orig, patch1, patch2 = ReadImages(
            DataPath)
        Img = np.array(patch_stack).reshape(1, patch_size, patch_size, 2)
        corner = np.array(src).reshape(1, 4, 2)
        patch1 = np.array(patch1).reshape(1, patch_size, patch_size, 1)
        patch2 = np.array(patch2).reshape(1, patch_size, patch_size, 1)
        FeedDict = {ImgPH: Img, CornerPH: corner,
                    patch1PH: patch1, patch2PH: patch2}
        res = sess.run(pred_H4pt, FeedDict)

    dst_pred = src + res.reshape(4,2)

    # cv2.polylines(img_orig, np.int32([src]), True, (0, 255, 0), 3)
    # cv2.polylines(img_orig, np.int32([dst]), True, (255, 0, 0), 5)
    # cv2.polylines(img_orig, np.int32([src_new]), True, (0, 0, 255), 5)
    # plt.figure()
    # plt.imshow(img_orig)
    # plt.show()
    H = cv2.getPerspectiveTransform(src, dst_pred)
    return H


def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    plt.figure(2)
    plt.imshow(img1)
    plt.figure(3)
    plt.imshow(img2)
    plt.show()
    return result


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/bernard/CMSC733/proj_1/Phase2/Checkpoints/',
                        help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', dest='BasePath', default='/home/bernard/CMSC733/proj_1/Phase2/Data/Val/',
                        help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt',
                        help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--ModelType', default='Unsup',
                        help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Args = Parser.parse_args()
    ModelType = Args.ModelType
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    ModelPath = Args.ModelPath + str(ModelType) + '/19model.ckpt'
    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)

    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, patch_size, patch_size, 2))
    if ModelType == 'Sup':
        H = TestSupervised(ImgPH, ImageSize, ModelPath, BasePath)
    elif ModelType == 'Unsup':
        
        H = TestUnsupervised(ImgPH, ImageSize, ModelPath, BasePath)
    else:
        print("ERROR: Unknown ModelType !!!")
        sys.exit()
    img_path = "/home/bernard/CMSC733/proj_1/Phase2/Data/P1TestSet/Phase1/TestSet3/"
    img1 = cv2.imread(img_path + '1.jpg')
    img2 = cv2.imread(img_path + '2.jpg')
    img1 = cv2.resize(img1, (320,240), interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (320,240), interpolation = cv2.INTER_AREA)
    H_inv = np.linalg.inv(H)
    res = warpTwoImages(img2, img1, H)
    cv2.imshow('r', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
