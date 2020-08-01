import sys
from processer import *
import numpy as np
import canton as ct
from canton import *
import tensorflow as tf
import time
import os
import math
import cv2
import  h5py
def mkdir(path):
    # 引入模块
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

def get_defense():
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


com,rec = ComCNN(),RecCNN()
com.summary()
rec.summary()

import time
from config import *

def compression_imagenet(adv_file_path="data/test_adv_PGD_0.00784.h5",save_com_file_path="data/test_com_PGD_0.00784.h5",threshold=.5):
    # get test data_adv
    h5_store = h5py.File(adv_file_path, 'r')
    train_data = h5_store['data'][:]  # 通过切片得到np数组
    train_target = h5_store['true_target'][:]
    h5_store.close()
    print("^_^ data loaded successfully from "+adv_file_path)

    #define batch info
    batch_size = 50
    nb_steps = 200

    # defence
    test = get_defense()
    get_session().run(ct.gvi())
    load()

    com_data = np.zeros(train_data.shape)
    for i in range(nb_steps):
        minibatch = train_data[i*batch_size:(i+1)*batch_size,:,:,:]
        minibatch = np.transpose(minibatch,[0,2,3,1])
        code, rec, code2, rec2,x= test(minibatch,threshold)
        rec  = np.transpose(rec,[0,3,1,2])
        com_data[i*batch_size:(i+1)*batch_size,:,:,:] = rec

    print("com_data:{}".format(com_data.shape))
    # save com_data
    h5_store = h5py.File(save_com_file_path,"w")
    h5_store.create_dataset('data',data=com_data)
    h5_store.create_dataset('target',data=train_target)
    h5_store.close()
    print("com_data saved in {} successfully ~".format(save_com_file_path))

def compression_cifar10(adv_file_path="data/test_adv_PGD_0.00784.h5",save_com_file_path="data/test_com_PGD_0.00784.h5",threshold=.5,set_size=10000):
    # get test data_adv
    h5_store = h5py.File(adv_file_path, 'r')
    train_data = h5_store['data'][:]  # 通过切片得到np数组
    try:
        train_target = h5_store['true_target'][:]
    except:
        train_target = h5_store['target'][:]
    h5_store.close()
    print("^_^ data loaded successfully from "+adv_file_path)

    #define batch info
    batch_size = 50
    nb_steps = set_size//batch_size

    # defence
    test = get_defense()
    get_session().run(ct.gvi())
    load()

    com_data = np.zeros(train_data.shape)
    for i in range(nb_steps):
        minibatch = train_data[i*batch_size:(i+1)*batch_size,:,:,:]
        minibatch = np.transpose(minibatch,[0,2,3,1])
        code, rec, code2, rec2,x= test(minibatch,threshold)
        rec  = np.transpose(rec,[0,3,1,2])
        com_data[i*batch_size:(i+1)*batch_size,:,:,:] = rec

    print("com_data:{}".format(com_data.shape))
    # save com_data
    h5_store = h5py.File(save_com_file_path,"w")
    h5_store.create_dataset('data',data=com_data)
    h5_store.create_dataset('target',data=train_target)
    h5_store.close()
    print("com_data saved in {} successfully ~".format(save_com_file_path))


if __name__ == '__main__':
    # for imagenet

    # prefix = "/home/Leeyegy/work_space/imagenet_adv/ImageNet_adv/data/"
    # adv_data_path = prefix + "test_ImageNet_"+str(args.set_size)+"_adv_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"
    # com_data_path = prefix + "test_ImageNet_"+str(args.set_size)+"_com_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"

    # for tiny_imagenet
    # prefix = "data/tiny_imagenet/"
    # adv_data_path = prefix + "test_tiny_ImageNet_1000_adv_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"
    # com_data_path = "data/test_tiny_ImageNet_1000_com_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"

    # for cifar10
    # prefix = "/home/Leeyegy/work_space/comdefend/Comdefend_tensorflow/data/"
    prefix="data/"
    adv_data_path = prefix + "test_adv_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"
    com_data_path = "data/test_com_" + str(args.attack_method) + "_" + str(args.epsilon) + ".h5"

    # for clean
    # adv_data_path = "data/test.h5"
    # com_data_path = "data/test_com.h5"

    # adv_data_path = prefix+"test_ImageNet_1000.h5"
    # com_data_path = prefix+"test_ImageNet_1000_com.h5"

    # if cifar10 : please use 10000 for set_size
    compression_cifar10(adv_file_path=adv_data_path,save_com_file_path=com_data_path,set_size=args.set_size)

    # test = get_defense()
    # get_session().run(ct.gvi())
    # load()
    # start = time.time()
    # # path='adv.png'
    # path='clean_image/6.png'
    #
    # path3="temp_imagenet/"
    # mkdir(path3)
    # Divided_Pach(path,path3)
    # for i in range(1,50):
    #         mkdir("temp_imagenet/")
    #         mkdir("com_imagenet_temp/")
    #         path1="temp_imagenet/"+str(i)+'.png'
    #         path2="com_imagenet_temp/"+str(i)+'.png'
    #         print(path2)
    #         Compression(path1,path2)
    # concated=mergeimage('com_imagenet_temp/')
    #     #print(concated.shape)
    # path4='result.png'
    # print(path4)
    # cv2.imwrite(path4,concated)
    # end = time.time()
# print(end-start)
