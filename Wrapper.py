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
    
    I1 = cv2.imread(BasePath)

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
    
    point1 = [rand_y, rand_x]
    point2 = [rand_y + patch_size, rand_x]
    point3 = [rand_y + patch_size, rand_x + patch_size]
    point4 = [rand_y, rand_x + patch_size]
    src = [point1, point2, point3, point4]
    src = np.array(src)
    # perform random perturbation in the range [-phi, phi] to the corner points of the patch
    dst = []
    for i in range(len(src)):
            rand_pertub = [src[i][0] + np.random.randint(-r, r), src[i][1]+np.random.randint(-r, r)]
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

def get_random_crop(img1, img2):
    img1 = cv2.imread(img1)
    img1 = np.float32(img1)
    img1 = cv2.resize(img1, (240, 320))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1_gray = (img1_gray-np.mean(img1_gray))/255

    img2 = cv2.imread(img2)
    img2 = np.float32(img2)
    img2 = cv2.resize(img2, (240, 320))
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_gray = (img2_gray-np.mean(img2_gray))/255

    rand_x = np.random.randint(r, img1.shape[1]-r-patch_size)
    rand_y = np.random.randint(r, img1.shape[0]-r-patch_size)
    
    point1 = [rand_y, rand_x]
    point2 = [rand_y + patch_size, rand_x]
    point3 = [rand_y + patch_size, rand_x + patch_size]
    point4 = [rand_y, rand_x + patch_size]
    rand_crop = [point1, point2, point3, point4]
    rand_crop = np.array(rand_crop)

    patch_1 = img1_gray[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
    patch_2 = img2_gray[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
    patch_stack = np.dstack((patch_1, patch_2))
    src = np.float32(rand_crop)
    patch_1 = np.float32(patch_1.reshape(patch_size, patch_size, 1))
    patch_2 = np.float32(patch_2.reshape(patch_size, patch_size, 1))
    return patch_stack, src, patch_1, patch_2

def get_middle_crop(img1, img2):
    img1 = cv2.imread(img1)
    img1 = np.float32(img1)
    img1 = cv2.resize(img1, (240, 320))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1_gray = (img1_gray-np.mean(img1_gray))/255

    img2 = cv2.imread(img2)
    img2 = np.float32(img2)
    img2 = cv2.resize(img2, (240, 320))
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2_gray = (img2_gray-np.mean(img2_gray))/255

    rand_x = int(120 - patch_size / 2)
    rand_y = int(160 - patch_size / 2)
    
    top_left = [rand_y, rand_x]
    top_right = [rand_y + patch_size, rand_x]
    bottom_right = [rand_y + patch_size, rand_x + patch_size]
    bottom_left = [rand_y, rand_x + patch_size]
    rand_crop = [top_left, top_right, bottom_right, bottom_left]
    rand_crop = np.array(rand_crop)

    patch_1 = img1_gray[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
    patch_2 = img2_gray[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
    patch_stack = np.dstack((patch_1, patch_2))
    src = np.float32(rand_crop)
    patch_1 = np.float32(patch_1.reshape(patch_size, patch_size, 1))
    patch_2 = np.float32(patch_2.reshape(patch_size, patch_size, 1))
    return patch_stack, src, patch_1, patch_2

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
    # patch_stack, H4pt, src, dst, img_orig, patch1, patch2 = ReadImages(DataPath)
    # Img = np.array(patch_stack).reshape(1, patch_size, patch_size, 2)
    # Predict output with forward pass, MiniBatchSize for Test is 1
    pred_H4pt = Supervised_HomographyModel(ImgPH, ImageSize, 1)

    # Setup Saver
    Saver = tf.train.Saver()

    with tf.Session() as sess:
        Saver.restore(sess, ModelPath)
        print('Number of parameters in this model are %d ' % np.sum(
            [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        H_list = []
        for i in tqdm(range(np.size(DataPath)-1)):
            H_mean = []
            # 4 random crop
            for j in range(5):
                if j != 4:
                    # patch_stack, H4pt, src, dst, img_orig, patch1, patch2 = ReadImages(DataPath[i])
                    patch_stack, src, patch1, patch2 = get_random_crop(DataPath[i+1], DataPath[i])
                else:
                    # make sure the fifth crop is in the middle
                    patch_stack, src, patch1, patch2 = get_middle_crop(DataPath[i+1], DataPath[i])

                Img = np.array(patch_stack).reshape(1, patch_size, patch_size, 2)
                FeedDict = {ImgPH: Img}
                res = sess.run(pred_H4pt, FeedDict)

                dst_pred = src + res.reshape(4,2)
                img = cv2.imread(DataPath[i+1])
                img = cv2.resize(img, (320, 240))
                cv2.polylines(img,np.int32([src]),True,(255,0,0), 3)
                cv2.polylines(img,np.int32([dst_pred]),True,(0,0,255), 5)
                plt.figure()
                plt.imshow(img)
                plt.show()
                H = cv2.getPerspectiveTransform(src, dst_pred)
                H_mean.append(H)
            
            H = np.mean(np.array(H_mean), axis=0)
            H_list.append(H)

    return H_list


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
        H_list = []
        for i in tqdm(range(np.size(DataPath)-1)):
            H_mean = []
            for j in range(5):
                if j != 4:
                    # patch_stack, H4pt, src, dst, img_orig, patch1, patch2 = ReadImages(DataPath[i])
                    patch_stack, src, patch1, patch2 = get_random_crop(DataPath[i+1], DataPath[i])
                else:
                    # make sure the fifth crop is in the middle
                    patch_stack, src, patch1, patch2 = get_middle_crop(DataPath[i+1], DataPath[i])
                Img = np.array(patch_stack).reshape(1, patch_size, patch_size, 2)
                corner = np.array(src).reshape(1, 4, 2)
                patch1 = np.array(patch1).reshape(1, patch_size, patch_size, 1)
                patch2 = np.array(patch2).reshape(1, patch_size, patch_size, 1)
                FeedDict = {ImgPH: Img, CornerPH: corner,
                            patch1PH: patch1, patch2PH: patch2}
                res = sess.run(pred_H4pt, FeedDict)

                dst_pred = src + res.reshape(4,2)
                # cv2.polylines(img2, np.int32([src]),True,(255,0,0), 3)
                # cv2.polylines(img2, np.int32([dst_pred]),True,(0,0,255), 5)
                # plt.figure()
                # plt.imshow(img2)
                # plt.show()
                H = cv2.getPerspectiveTransform(src, dst_pred)
                H_mean.append(H)
            H = np.mean(np.array(H_mean), axis=0)
            H_list.append(H)

    return H_list


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
    plt.figure(1)
    plt.imshow(result)
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    # plt.figure(2)
    # plt.imshow(img1)
    # plt.figure(3)
    # plt.imshow(result)
    plt.show()
    return result

# Displays image
def visualize_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, interpolation='nearest')
    plt.axis('off')
    plt.show()

def apply_homography(H, src):
    """ Apply the homography H to src
    Parameters:
    - H: the 3x3 homography matrix
    - src: (x,y) coordinates of N source pixels, where coordinates are row vectors,
           so the matrix has dimension Ndest[:, 0] (N>=4).
    Returns:
    - src: (x,y) coordinates of N destination pixels, where coordinates are row vectors,
           so the matrix has dimension Ndest[:, 0] (N>=4).
    Author:
    - Yu Fang
    """
    src_h = np.hstack((src, np.ones((src.shape[0], 1))))
    dest = src_h @ H.T
    return (dest / dest[:, [2]]+0.001)[:, 0:2]

def stitch(image1, image2, H):
    # Find height and width of both images
    x1, y1 = tuple(image1.shape[:2])
    x2, y2 = tuple(image2.shape[:2])
    # Find estimated location of bounding box
    top_left = apply_homography(H, np.array([[0, 0]]))[0]
    top_right = apply_homography(H, np.array([[0, y1]]))[0]
    bottom_left = apply_homography(H, np.array([[x1, 0]]))[0]
    bottom_right = apply_homography(H, np.array([[x1, y1]]))[0]
    # Determine left and right bounds to calculate width of stitched image
    left_bound = np.amin([top_left[0], bottom_left[0], 0])
    right_bound = np.amax([top_right[0], bottom_right[0], x2])
    y = int(right_bound - left_bound)
    # Similarly, determine upper and lower bounds to calculate height of stitched image
    upper_bound = np.amin([top_left[1], top_right[1], 0])
    lower_bound = np.amax([bottom_left[1], bottom_right[1], y2])
    x = int(lower_bound - upper_bound)
    # Delta values of x and y for translation matrix
    delta_x = -int(left_bound)
    delta_y = -int(upper_bound)
    # Define translation matrix to move upper left corner to (0,0)
    T = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])
    # Warp first image to perspective of second image
    # Swapped axes for correct result
    result = np.swapaxes(cv2.warpPerspective(np.swapaxes(image1, 0, 1), T @ H, (y, x)), 0, 1)
    visualize_image(result)
    # Find overlap region using delta values
    overlap = result[delta_x:x2+delta_x, delta_y:y2+delta_y]
    # Blending using average pixel value
    overlap = ((image2.astype(int)+overlap.astype(int))/2).astype(np.uint8)
    # Overlap image to create stitched image
    result[delta_x:x2+delta_x, delta_y:y2+delta_y] = overlap
    return result

def stitchImagePairs(img0, img1, H):
    '''stitch image 0 on image 1'''
    image0 = img0.copy()
    image1 = img1.copy()

    print("shapes")
    print(image0.shape)
    print(image1.shape)
    

    h0 ,w0 = image0.shape[:2]
    h1 ,w1 = image1.shape[:2]

    points_on_image0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1,1,2)
    points_on_image0_transformed = cv2.perspectiveTransform(points_on_image0, H)
    print("transformed points = ", points_on_image0_transformed)
    points_on_image1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1,1,2)

    points_on_merged_images = np.concatenate((points_on_image0_transformed, points_on_image1), axis = 0)
    points_on_merged_images_ = []

    for p in range(len(points_on_merged_images)):
        points_on_merged_images_.append(points_on_merged_images[p].ravel())

    points_on_merged_images_ = np.array(points_on_merged_images_)

    # x_min, y_min = np.int0(np.min(points_on_merged_images_, axis = 0))
    # x_max, y_max = np.int0(np.max(points_on_merged_images_, axis = 0))
    [x_min, y_min] = np.int32(points_on_merged_images_.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points_on_merged_images_.max(axis=0).ravel() + 0.5)
    print("min, max")
    print(x_min, y_min)
    print(x_max, y_max)

    overlap_area = cv2.polylines(image1,[np.int32(points_on_image0_transformed)],True,255,3, cv2.LINE_AA) 
    cv2.imshow("original_image_overlapping.jpg", overlap_area)
    cv2.waitKey() 
    cv2.destroyAllWindows()
    H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # translate

    image0_transformed_and_stitched = cv2.warpPerspective(image0, H_translate @ H, (x_max-x_min, y_max-y_min))
    visualize_image(image0_transformed_and_stitched)
    image0_transformed_and_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = image1
    return image0_transformed_and_stitched


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
    Parser.add_argument('--BasePath', dest='BasePath', default="/home/bernard/CMSC733/proj_1/Phase1/Data/Train/Set1/",
                        help='Path to load images from, Default:BasePath')
    Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt',
                        help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--ModelType', default='Sup',
                        help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Args = Parser.parse_args()
    ModelType = Args.ModelType
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    ModelPath = Args.ModelPath + str(ModelType) + '/29model.ckpt'
    # Setup all needed parameters including file reading
    ImageSize, DataPath = SetupAll(BasePath)
    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder('float', shape=(1, patch_size, patch_size, 2))
    # img_path = "/home/bernard/CMSC733/proj_1/Phase2/Data/P1TestSet/Phase1/TestSet3/"
    img_path = "/home/bernard/CMSC733/proj_1/Phase1/Data/Train/Set1/"
    img1 = cv2.imread(img_path + '1.jpg')
    # img2 = cv2.imread(img_path + '2.jpg')
    # img1 = cv2.imread(DataPath[0])
    res = cv2.resize(img1, (320, 240), interpolation = cv2.INTER_AREA)
    if ModelType == 'Sup':
        H_list = TestSupervised(ImgPH, ImageSize, ModelPath, DataPath)
            
    elif ModelType == 'Unsup':
        H_list = TestUnsupervised(ImgPH, ImageSize, ModelPath, DataPath)
    else:
        print("ERROR: Unknown ModelType !!!")
        sys.exit()
    NumImages = len(glob.glob(img_path+'*.jpg'))
    for i in range(1, NumImages):
        img2 = cv2.imread(img_path + str(i+1) + '.jpg')
        img2 = cv2.resize(img2, (320, 240), interpolation = cv2.INTER_AREA)
        H_inv = np.linalg.inv(H_list[i-1])
        res = warpTwoImages(res, img2, H_list[i - 1])
        # res = stitchImagePairs(img2, res, H_list[i-1])
        # res = stitch(res, img2, H_inv)
        cv2.imshow('r', res)
        cv2.waitKey(0)
    
    cv2.imshow('r2', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
