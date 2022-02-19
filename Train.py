#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import Supervised_HomographyModel, Unsupervised_HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
# from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

if (len(tf.config.experimental.list_physical_devices('GPU')) > 0):
    print("RUNNING in GPU !!!")
else:
    print("NO GPU FOUND !!!")

# Don't generate pyc codes
sys.dont_write_bytecode = True

patch_size = 128
# generate random patch (square)
r = 32


def getPatchIndices(corners_a):
    patch_indices = []
    for i in range(corners_a.shape[0]):
        xmin, ymin = corners_a[i, 0, 0], corners_a[i, 0, 1]
        xmax, ymax = corners_a[i, 3, 0], corners_a[i, 3, 1]
#         print(xmin,ymin,xmax,ymax)
        X, Y = np.mgrid[xmin:xmax, ymin:ymax]
        patch_indices.append(np.dstack((Y, X)))
    return np.array(patch_indices)


def GenerateBatch(BasePath, DirNamesTrain, ImageSize, MiniBatchSize, ModelType):
    """
    Inputs: 
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    StackBatch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """

    StackBatch = []
    LabelBatch = []
    CornerBatch = []
    patch1Batch = []
    patch2Batch = []
    imgs = []
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)

        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'
        ImageNum += 1

        ##########################################################
        # Add any standardization or data augmentation here!
        ##########################################################
        img = np.float32(cv2.imread(RandImageName))
        img = cv2.resize(img, (240, 320))
        # I1 = img
        # if(ImageSize[2] == 3):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_gray = np.float32(img_gray)
        # I1 = (I1 - np.mean(I1))/255

        img_gray = (img_gray-np.mean(img_gray))/255

        # rand_x = np.random.randint(r, img_gray.shape[1]-r-patch_size)
        # rand_y = np.random.randint(r, img_gray.shape[0]-r-patch_size)
        if(img.shape[1]-r-patch_size) > r+1 and (img.shape[0]-r-patch_size) > r+1:
            rand_x = np.random.randint(r, img.shape[1]-r-patch_size)
            rand_y = np.random.randint(r, img.shape[0]-r-patch_size)
        else:
            ImageNum = ImageNum - 1
            continue

        point1 = [rand_y, rand_x]
        point2 = [rand_y + patch_size, rand_x]
        point3 = [rand_y + patch_size, rand_x + patch_size]
        point4 = [rand_y, rand_x + patch_size]
        src = [point1, point2, point3, point4]
        src = np.array(src)
        # perform random perturbation in the range [-phi, phi] to the corner points of the patch
        theta = 2 * np.pi * random.random()
        dst = []
        for i in range(len(src)):
            rand_pertub = [src[i][0] + np.random.randint(-r, r), src[i][1]+np.random.randint(-r, r)]
            dst.append(rand_pertub)

        H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        H_inv = np.linalg.inv(H)
        img_warp = cv2.warpPerspective(
            img_gray, H_inv, (img_gray.shape[1], img_gray.shape[0]))
        patch_1 = img_gray[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
        # patch_2 = img_warp[rand_y:rand_y+patch_size, rand_x:rand_x+patch_size]
        patch_2 = img_warp[src[0][0]:src[2][0], src[0][1]:src[2][1]]
        # patch_1 = (patch_1 - np.mean(patch_1))/255
        # patch_2 = (patch_2 - np.mean(patch_2))/255
        patch_stack = np.dstack((patch_1, patch_2))
        H4pt = np.subtract(dst, src)
        StackBatch.append(patch_stack)
        LabelBatch.append(H4pt.reshape(8,))
        patch1Batch.append(np.float32(
            patch_1.reshape(patch_size, patch_size, 1)))
        CornerBatch.append(np.float32(src))
        patch2Batch.append(np.float32(
            patch_2.reshape(patch_size, patch_size, 1)))

        imgs.append(img_gray.reshape(img_gray.shape[1], img_gray.shape[0], 1))
    # patch_indices = getPatchIndices(src)
    if ModelType == 'Sup':
        return StackBatch, LabelBatch
    elif ModelType == 'Unsup':
        return StackBatch, CornerBatch, patch1Batch, patch2Batch, imgs


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)


def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    if ModelType == 'Sup':
        H4pt = Supervised_HomographyModel(ImgPH, ImageSize, MiniBatchSize)
    elif ModelType == 'Unsup':
        CornerPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, 4, 2))
        I1PH = tf.placeholder(tf.float32, shape=(
            MiniBatchSize, patch_size, patch_size, 1))
        I2PH = tf.placeholder(tf.float32, shape=(
            MiniBatchSize, patch_size, patch_size, 1))
        imgPH = tf.placeholder(tf.float32, shape=(
            MiniBatchSize, 320, 240, 1))
        patchIndicesPH = tf.placeholder(tf.int32, shape=(
            MiniBatchSize, patch_size, patch_size, 2))
        pred_I2, patch2, H4pt = Unsupervised_HomographyModel(
            ImgPH, CornerPH, I1PH, I2PH, ImageSize, MiniBatchSize)
    else:
        print('ERROR: Unknown ModelType !!!')
        sys.exit()

    with tf.name_scope('Loss'):
        ###############################################
        # Fill your loss function of choice here!
        ###############################################
        if ModelType == 'Sup':
            # L2-loss
            loss = tf.sqrt(tf.reduce_sum(
                (tf.squared_difference(H4pt, LabelPH))))
        elif ModelType == 'Unsup':
            loss = tf.reduce_mean(tf.abs(pred_I2 - patch2))

    with tf.name_scope('Adam'):
        ###############################################
        # Fill your optimizer of choice here!
        ###############################################
        # if ModelType == 'Sup':
        #     Optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        # elif ModelType == 'Unsup':
        #     Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Tensorboard
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    # tf.summary.image('Anything you want', AnyImg)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    # loss_summary = tf.summary.scalar('LossEveryIter', loss)
    # MergedSummaryOP1 = tf.summary.merge([loss_summary])
    # Setup Saver
    Saver = tf.train.Saver()
    LossOverEpochs = np.array([0, 0])
    with tf.Session() as sess:
        if LatestFile is not None:
            Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
            # Extract only numbers from the name
            StartEpoch = int(
                ''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
            print('Loaded latest checkpoint with the name ' + LatestFile + '....')
        else:
            sess.run(tf.global_variables_initializer())
            StartEpoch = 0
            print('New model initialized....')

        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
        if ModelType == 'Sup':
            for Epochs in tqdm(range(StartEpoch, NumEpochs)):
                NumIterationsPerEpoch = int(
                    NumTrainSamples/MiniBatchSize/DivTrain)
                appendLoss = []
                for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                    StackBatch, LabelBatch = GenerateBatch(
                        BasePath, DirNamesTrain, ImageSize, MiniBatchSize, ModelType)
                    FeedDict = {ImgPH: StackBatch, LabelPH: LabelBatch}
                    _, LossThisBatch, Summary = sess.run(
                        [Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                    appendLoss.append(LossThisBatch)
                    # Tensorboard
                    Writer.add_summary(
                        Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                    # If you don't flush the tensorboard doesn't update until a lot of iterations!
                    Writer.flush()

                # Save model every epoch
                SaveName = CheckPointPath + ModelType + \
                    '/' + str(Epochs) + 'model.ckpt'
                Saver.save(sess, save_path=SaveName)
                print('\n' + SaveName + ' Model Saved...')
                print("Loss: ", np.mean(appendLoss))
                LossOverEpochs = np.vstack(
                    (LossOverEpochs, [Epochs, np.mean(appendLoss)]))
                tf.summary.scalar('LossOverEpochs', np.mean(appendLoss))
            plt.xlim(0, 30)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(LossOverEpochs[:, 0], LossOverEpochs[:, 1])
            if not os.path.exists('Graphs/Sup/train/'):
                os.makedirs('Graphs/Sup/train/')
            plt.savefig('Graphs/Sup/train/lossEpochs'+str(Epochs)+'.png')
            # plt.show()
            plt.close()

        elif ModelType == 'Unsup':
            for Epochs in tqdm(range(StartEpoch, NumEpochs)):
                NumIterationsPerEpoch = int(
                    NumTrainSamples/MiniBatchSize/DivTrain)
                appendLoss = []
                for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                    StackBatch, CornerBatch, patch1Batch, patch2Batch, imgs = GenerateBatch(
                        BasePath, DirNamesTrain, ImageSize, MiniBatchSize, ModelType)
                    FeedDict = {ImgPH: StackBatch, CornerPH: CornerBatch,
                                I1PH: patch1Batch, I2PH: patch2Batch}
                    # FeedDict = {ImgPH: StackBatch, CornerPH: CornerBatch, I2PH: patch2Batch}
                    _, LossThisBatch, Summary = sess.run(
                        [Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)

                    appendLoss.append(LossThisBatch)
                    # Tensorboard
                    Writer.add_summary(
                        Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                    # If you don't flush the tensorboard doesn't update until a lot of iterations!
                    Writer.flush()

                # Save model every epoch
                SaveName = CheckPointPath + ModelType + \
                    '/' + str(Epochs) + 'model.ckpt'
                Saver.save(sess, save_path=SaveName)
                print('\n' + SaveName + ' Model Saved...')
                print("Loss: ", np.mean(appendLoss))
                LossOverEpochs = np.vstack(
                    (LossOverEpochs, [Epochs, np.mean(appendLoss)]))
                tf.summary.scalar('LossOverEpochs', np.mean(appendLoss))
            plt.xlim(0, 30)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(LossOverEpochs[:, 0], LossOverEpochs[:, 1])
            if not os.path.exists('Graphs/Unsup/train/'):
                os.makedirs('Graphs/Unsup/train/')
            plt.savefig('Graphs/Unsup/train/lossEpochs'+str(Epochs)+'.png')
            # plt.show()
            plt.close()


def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/bernard/CMSC733/proj_1/Phase2/Data/',
                        help='Base path of images, Default:/home/bernard/CMSC733/proj_1/Phase2/Data/')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/',
                        help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Sup',
                        help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=30,
                        help='Number of Epochs to Train for, Default:30')
    Parser.add_argument('--DivTrain', type=int, default=1,
                        help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=64,
                        help='Size of the MiniBatch to use, Default:16')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0,
                        help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/',
                        help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(
        BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    # PrettyPrint(NumEpochs, DivTrain, MiniBatchSize,
    #             NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    tf.compat.v1.disable_eager_execution()
    ImgPH = tf.compat.v1.placeholder(tf.float32, shape=(
        MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    LabelPH = tf.compat.v1.placeholder(tf.float32, shape=(
        MiniBatchSize, NumClasses))  # OneHOT labels

    TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType)


if __name__ == '__main__':
    main()
