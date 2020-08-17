from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from skimage.io import imread, imsave
from skimage.measure import compare_psnr, compare_ssim
import cv2

import numpy as np
import matplotlib
import  time

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import os, glob, datetime, time
import sys
from data_generator import get_handled_cifar10_test_loader
from networks import  *
torch.multiprocessing.set_sharing_strategy('file_system')


import  os
from config import  args

def ssim_and_save(com_data,cln_data):
    '''
    :param com_data: [N,C,H,W] | np.array [0,1]
    :param cln_data: [N,C,H,W] | np.array [0,1]
    :return:
    '''
    com_data = ((np.transpose(com_data.numpy(),[0,2,3,1]))*255).astype(np.float32)
    cln_data = ((np.transpose(cln_data,[0,2,3,1]))*255).astype(np.float32) # only np.float32 is supported

    for index in range(com_data.shape[0]):
        com_img = com_data[index]
        cln_img = cln_data[index]
        print(com_img.shape)
        grayA = cv2.cvtColor(com_img, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(cln_img, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        print("img index: {} , SSIM: {}".format(index,score))
        com_filename = os.path.join("defend_image",str(index)+".png")
        cln_filename = os.path.join("clean_image",str(index)+".png")
        cv2.imwrite(com_filename,com_data[index])
        cv2.imwrite(cln_filename,cln_data[index])


if __name__ == '__main__':
    torch.manual_seed(0)
    # check file
    # imagenet
    # prefix = "/home/Leeyegy/work_space/imagenet_adv/ImageNet_adv/data/"
    # com_data_path = prefix + "test_ImageNet_"+str(args.set_size)+"_com_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"

    # #tiny_imagenet
    # com_data_path = os.path.join("data/tiny_imagenet","test_tiny_ImageNet_1000_com_"+str(args.attack_method)+"_"+str(args.epsilon)+".h5")


    # cifar10
    com_data_path = os.path.join("data","test_com_"+str(args.attack_method)+"_"+str(args.epsilon)+".h5")
    # com_data_path = os.path.join("data","test_adv_"+str(args.attack_method)+"_"+str(args.epsilon)+".h5")

    # clean
    # com_data_path = os.path.join("data","test_com.h5")

    # com_data_path = prefix + "test_ImageNet_"+str(args.set_size)+"_com_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"
    assert os.path.exists(com_data_path), "not found expected file : "+com_data_path

    # if not os.path.exists(com_data_path):
    #     compression_cifar10(adv_file_path=adv_data_path,save_com_file_path=com_data_path)

    # get cln data and restored data
    import  h5py
    h5_store = h5py.File(com_data_path,"r")
    com_data = torch.from_numpy(h5_store['data'][:])
    h5_store.close()

    h5_store = h5py.File("data/test.h5", 'r')
    cln_data = h5_store['data'][:]
    h5_store.close()

    #define batch_size
    batch_size = 50
    nb_steps=args.set_size // batch_size

    for i in range(nb_steps):
        com_data_batch = com_data[i*batch_size:(i+1)*batch_size,:,:,:]
        cln_data_batch = cln_data[i*batch_size:(i+1)*batch_size,:,:,:]
        ssim_and_save(com_data_batch,cln_data_batch)





