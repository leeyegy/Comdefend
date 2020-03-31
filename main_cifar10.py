# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


'''
test for the three adding steps:
1. using low-level gaussian noise
2. incorporating the add-de-noise into training model
3. using low-level adversarial noise
'''


from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import CarliniWagnerL2Attack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack

from skimage.io import imread, imsave
from skimage.measure import compare_psnr, compare_ssim


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

from networks import  *
# sys.path.append("../../")
# from utils_net.net.networks import *


torch.multiprocessing.set_sharing_strategy('file_system')


import  os
from config import  args

if __name__ == '__main__':
    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # check file
    # imagenet
    # prefix = "/home/Leeyegy/work_space/imagenet_adv/ImageNet_adv/data/"
    # com_data_path = prefix + "test_ImageNet_"+str(args.set_size)+"_com_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"

    # cifar10
    com_data_path = os.path.join("data","test_com_"+str(args.attack_method)+"_"+str(args.epsilon)+".h5")

    # com_data_path = prefix + "test_ImageNet_"+str(args.set_size)+"_com_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"
    assert os.path.exists(com_data_path), "not found expected file : "+com_data_path

    # if not os.path.exists(com_data_path):
    #     compression_cifar10(adv_file_path=adv_data_path,save_com_file_path=com_data_path)



    # get test data
    import  h5py
    h5_store = h5py.File(com_data_path,"r")
    com_data = torch.from_numpy(h5_store['data'][:])
    true_target = torch.from_numpy(h5_store['target'][:])
    h5_store.close()

    #define batch_size
    batch_size = 50
    nb_steps=args.set_size // batch_size

    #load net
    print('| Resuming from checkpoints...')
    assert os.path.isdir('checkpoints'), 'Error: No checkpoint directory found!'
    checkpoint = torch.load('./checkpoints/wide-resnet-28x10.t7') #
    model = checkpoint['net']
    # model = torch.load('./checkpoints/resnet50_epoch_7.pth')
    nb_epoch = 1


    model = model.to(device)
    for epoch in range(nb_epoch):
        #evaluate
        model.eval()
        clncorrect = 0

        for i in range(nb_steps):
            print("{}/{}".format(i,nb_steps))
            clndata = com_data[i*batch_size:(i+1)*batch_size,:,:,:].to(device)
            target = true_target[i*batch_size:(i+1)*batch_size].to(device)
            # clndata, target = torch.from_numpy(com_data[i*batch_size:(i+1)*batch_size,:,:,:]).to(device), target[i*batch_size:(i+1)*batch_size].to(device)
            # print("target:{}".format(target.size()))
            with torch.no_grad():
                output = model(clndata.float())

            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()


        print('\nTest set with defence: '
              ' cln acc: {}/{} ({:.0f}%)\n'.format( clncorrect, args.set_size,
                  100. * clncorrect / args.set_size))


