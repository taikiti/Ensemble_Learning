import sys
import argparse
import chainer
import numpy as np
import pickle
import matplotlib.pyplot as plt
from chainer import cuda,optimizers
from chainer import Variable,Chain,dataset, datasets
from chainer import serializers
from convolution import CNNetwork
import os
import cv2

from chainer.datasets import mnist
from chainer.datasets import split_dataset_random
from chainer import iterators
import random
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
import matplotlib as mpl
import chainer.functions as F

plt.switch_backend('agg')

def get_data():
    np.random.seed(0)
    os.system('rm -r `find test -name .DS_Store`')
    print("get data now...")
    srcDir = './dataset'
    user = os.listdir(srcDir)

    image_num = 0
    for u in user:
        if u!='.DS_Store':
            n = len(os.listdir(srcDir+"/"+str(u)))
            print(u,n)
            image_num += n
    print('image_num =',image_num)
    dimX = 500
    dimY = 100
    dim = dimX * dimY
    data = np.zeros(image_num * dim, dtype=np.float32).reshape((image_num, dim))
    target = np.zeros(image_num, dtype=np.int32).reshape((image_num, ))

    ct = 0
    for (i, u) in enumerate(user):
        print(i,u)
        if u!='.DS_Store':
             #debag
            for (j, f) in enumerate(os.listdir(srcDir + "/" + u + '/')):
                 if f !='.DS_Store':
                        img = cv2.imread(srcDir +"/" + u + '/' + f, cv2.IMREAD_GRAYSCALE)
                        #print("file->",srcDir + "/"+u + '/' + f)  #dabag
                        img = cv2.resize(img, (dimX,dimY))
                        #deleteDegree(img, dimX, dimY)
                        img2 = img
                        img = img / 255
                        img3 = img
                        trg = img
                        img = [flatten for inner in img for flatten in inner]
                        for c in range(dim):
                            data[ct,c] = img[c]
                        target[ct] =  (int(f[-6])*10+int(f[-5]))%14
                        ct += 1
    data2 = np.zeros(image_num * dim, dtype=np.float32).reshape((image_num, dim))
    target2 = np.zeros(image_num, dtype=np.int32).reshape((image_num, ))

    import random
    indexlist = list(range(ct))
    random.shuffle(indexlist)

    for i in range(ct):
        for c in range(dim):
            data2[i, c] = data[indexlist[i], c]
        target2[i] = target[indexlist[i]]

    data3 = data2.astype(np.float32)
    label = target2.astype(np.int32)
    N = 1500  # of training data
    N_test = 547  # of test data

    train_data = data3[:N].reshape((N, 1, 100, 500))
    test_data = data3[N:].reshape((N_test, 1, 100, 500))
    train_label = label[:N]
    test_label = label[N:]
    print("...read OK")
    return train_data, train_label, test_data, test_label
def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
def get_model_optimizer(args):
    model = CNNetwork()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    if args.optimizer == 'SGD':
        optimizer = optimizers.MomentumSGD(lr=args.lr,momentum=args.momentum)
    optimizer.setup(model)
    return model,optimizer

if __name__ == '__main__':

    reset_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.001, type=float)
    parser.add_argument('--power', default=0.75, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str,
                        choices=['SGD', 'Adam'])
    args = parser.parse_args()


    train_data, train_label, test_data, test_label = get_data()
    trainT = datasets.TupleDataset(train_data, train_label)
    test =  datasets.TupleDataset(test_data, test_label)
    train, valid = split_dataset_random(trainT,1000,seed=0)

    gpu_id = -1  # CPUを用いる場合は、この値を-1にしてください
    batchsize=64
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    model,optimizer = get_model_optimizer(args)

    train_iter = iterators.SerialIterator(train,batchsize)
    valid_iter = iterators.SerialIterator(valid,batchsize,repeat=False,shuffle=False)
    test_iter = iterators.SerialIterator(test,batchsize,repeat=False,shuffle=False)

    max_epoch = 100
    mean_loss=0
    delta = 1e-7
    while train_iter.epoch < max_epoch:
        train_batch = train_iter.next()
        x,t =concat_examples(train_batch,gpu_id)

        y = model(x)
        loss = F.softmax_cross_entropy(y+delta, t)
         # 勾配の計算
        model.cleargrads()
        loss.backward()
        # パラメータの更新
        optimizer.update()
        #ここまでで１epoch
        if train_iter.is_new_epoch:
            print('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))))

            valid_losses = []
            valid_accuracies = []
            while True:
                valid_batch = valid_iter.next()
                x_valid, t_valid = concat_examples(valid_batch, gpu_id)
                with chainer.using_config('train', False),chainer.using_config('enable_backprop', False):
                    y_valid = model(x_valid)
                loss_valid = F.softmax_cross_entropy(y_valid+delta, t_valid)
                valid_losses.append(to_cpu(loss_valid.array))
                accuracy = F.accuracy(y_valid, t_valid)
                accuracy.to_cpu()
                valid_accuracies.append(accuracy.array)

                if valid_iter.is_new_epoch:
                    valid_iter.reset()
                    break
            print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(valid_losses), np.mean(valid_accuracies)))

        test_accuracies = []
    while True:
        test_batch = test_iter.next()
        x_test, t_test = concat_examples(test_batch, gpu_id)

            # テストデータをforward
        with chainer.using_config('train', False),chainer.using_config('enable_backprop', False):
            y_test = model(x_test)

        # 精度を計算
        accuracy = F.accuracy(y_test, t_test)
        accuracy.to_cpu()
        test_accuracies.append(accuracy.array)

        if test_iter.is_new_epoch:
            test_iter.reset()
            break

    print('test_accuracy:{:.04f}'.format(np.mean(test_accuracies)))


            #空白
