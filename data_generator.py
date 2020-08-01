import glob
import cv2
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import h5py
import argparse

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import CarliniWagnerL2Attack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CIFAR10Dataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, data, target):
        super(CIFAR10Dataset, self).__init__()
        self.data = data
        self.target = target

    def __getitem__(self, index): # 该函数涉及到enumerate的返回值
        batch_x = self.data[index]
        batch_y = self.target[index]
        return batch_x, batch_y

    def __len__(self):
        return self.data.size(0)

def get_raw_cifar10_data(loader):
    train_data = []
    train_target = []

    for batch_idx, (data, target) in enumerate(loader):
        train_data.append(data.numpy())
        train_target.append(target.numpy())

    train_data = np.asarray(train_data)
    train_target = np.asarray(train_target)
    train_data = train_data.reshape([-1, 3, 32, 32])
    train_target = np.reshape(train_target, [-1])

    return train_data, train_target

def get_handled_cifar10_test_loader(batch_size, num_workers, shuffle=True):
    if os.path.exists("data/test.h5"):
        h5_store = h5py.File("data/test.h5", 'r')
        train_data = h5_store['data'][:]
        train_target = h5_store['target'][:]
        h5_store.close()
        print("^_^ data loaded successfully from test.h5")

    else:
        h5_store = h5py.File("data/test.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()


    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)

    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=num_workers, drop_last=True, batch_size=batch_size,
                      shuffle=shuffle)

