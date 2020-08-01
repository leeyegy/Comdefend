from BPDA import  BPDAattack
import  numpy as np
import torch
import argparse
from data_generator import  get_handled_cifar10_test_loader
import os
import sys
from processer import *
import numpy as np
import canton as ct
from canton import *
import tensorflow as tf
import time
import math
import cv2
import  h5py
def mkdir(path):
    import os
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('make')
def ComCNN():
    c=Can()
    def conv(nip,nop,flag=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if flag:
            c.add(Act('elu'))
    c.add(Lambda(lambda x:x-0.5))
    conv(3,16)
    conv(16,32)
    conv(32,64)
    conv(64,128)
    conv(128,256)
    conv(256,128)
    conv(128,64)
    conv(64,32)
    conv(32,12,flag=False)
    c.chain()
    return c

def RecCNN():
    c=Can()
    def conv(nip,nop,flag=True):
        c.add(Conv2D(nip,nop,k=3,usebias=True))
        if flag:
            c.add(Act('elu'))
    conv(12,32)
    conv(32,64)
    conv(64,128)
    conv(128,256)
    conv(256,128)
    conv(128,64)
    conv(64,32)
    conv(32,16)
    conv(16,3,flag=False)
    c.add(Act('sigmoid'))
    c.chain()
    return c

with tf.device('/gpu:2'):
    com, rec = ComCNN(), RecCNN()
    com.summary()
    rec.summary()

def get_defense():
    with tf.device('/gpu:2'):
        x = ph([None,None,3])
        x = tf.clip_by_value(x,clip_value_max=1.,clip_value_min=0.)
        code_noise = tf.Variable(1.0)
        linear_code = com(x)
        noisy_code = linear_code - \
            tf.random_normal(stddev=code_noise,shape=tf.shape(linear_code))
        binary_code = Act('sigmoid')(noisy_code)
        y = rec(binary_code)
        set_training_state(False)
        quantization_threshold = tf.Variable(0.5)
        binary_code_test = tf.cast(binary_code>quantization_threshold,tf.float32)
        y_test = rec(binary_code_test)
    def test(batch,quanth):
        with tf.device('/gpu:2'):
            sess = ct.get_session()
            res = sess.run([binary_code_test,y_test,binary_code,y,x],feed_dict={
                x:batch,
                quantization_threshold:quanth,
            })
        return res
    return test

def Compression(path,path1,threshold=.5): #将路径中图像压缩还原并保存再路径中 code bool code2 float
    import cv2
    image=readimage(path)
    minibatch =[image]
    minibatch=np.array(minibatch)
    print(minibatch.shape)
    code, rec, code2, rec2,x= test(minibatch,threshold)
    img2=change(rec)
    cv2.imwrite(path1,img2)
    return code, code2

def load():
    print("****")
    com.load_weights('checkpoints/enc20_0.0001.npy')
    rec.load_weights('checkpoints/dec20_0.0001.npy')

def defence(data,test):
    '''
    :param data: tensor.cuda() | [N,C,H,W] | [0,1]
    :return: tensor.cuda() | [N,C,H,W] | [0,1]
    '''
    # defence
    code, rec, code2, rec2, x = test(np.transpose(data.cpu().numpy(), [0, 2, 3, 1]), 0.5)
    return torch.from_numpy(np.transpose(rec, [0, 3, 1, 2])).cuda()

def main(args):
    # load data
    testLoader = get_handled_cifar10_test_loader(batch_size=50, num_workers=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    print('| Resuming from checkpoints...')
    assert os.path.isdir('checkpoints'), 'Error: No checkpoint directory found!'
    checkpoint = torch.load('./checkpoints/wide-resnet-28x10.t7') # for cifar10
    model = checkpoint['net']
    # model = torch.nn.DataParallel(model)
    model = model.to(device)

    # defence
    test = get_defense()

    with tf.device('/gpu:2'):
        get_session().run(ct.gvi())
        load()

    # define adversary
    adversary = BPDAattack(model, defence, device,
                                epsilon=args.epsilon,
                                learning_rate=0.01,
                                max_iterations=args.max_iterations,test=test)

    # model test
    model.eval()
    clncorrect_nodefence = 0
    for data,target in testLoader:
        data, target = data.cuda(0), target.cuda(0)
        # attack
        adv_data = adversary.perturb(data,target)
        # defence
        denoised_data = defence(adv_data,test).cuda(0)
        with torch.no_grad():
            output = model(denoised_data.float())
        pred = output.max(1, keepdim=True)[1]
        pred = pred.double()
        target = target.double()
        clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()  # item： to get the value of tensor
    print('\nTest set with feature-dis defence against BPDA'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                clncorrect_nodefence, len(testLoader.dataset),
                  100. * clncorrect_nodefence / len(testLoader.dataset)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--epsilon",type=float,default=8/255)

    # BPDA ATTACK
    parser.add_argument("--max_iterations",default=10,type =int)
    args = parser.parse_args()
    main(args)