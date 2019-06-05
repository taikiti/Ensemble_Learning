import sys
import argparse
import chainer
import numpy as np
import pickle
import matplotlib.pyplot as plt
from chainer import cuda,optimizers
from chainer import Variable,Chain,dataset, datasets
from chainer import serializers
from convolution import CNNetwork9
import os
import cv2
from chainer.datasets import mnist
from chainer.datasets import split_dataset_random
from chainer import iterators
from chainer import training
from chainer.training import extensions

import random
from chainer.dataset import concat_examples
from chainer.cuda import to_cpu
import matplotlib as mpl
import chainer.functions as F
import chainer.links as L
plt.switch_backend('agg')

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
def get_model_optimizer(args):
    model = CNNetwork9()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    if args.optimizer == 'SGD':
        optimizer = optimizers.MomentumSGD(lr=args.lr,momentum=args.momentum)

    return model,optimizer

if __name__ == '__main__':

    # reset_seed(0)
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

    import pickle
    train_pickle=pickle.load(open('train.dat','rb'),encoding = 'bytes')
    test_pickle=pickle.load(open('test.dat','rb'),encoding = 'bytes')
    train_data = train_pickle['data']
    train_label = train_pickle['target']
    test_data = test_pickle['data']
    test_label = test_pickle['target']

    trainT = datasets.TupleDataset(train_data, train_label)
    test =  datasets.TupleDataset(test_data, test_label)
    train, valid = split_dataset_random(trainT,1000,seed=0)

    gpu_id = 0  # CPUを用いる場合は、この値を-1にしてください
    batchsize=64
    model,optimizer = get_model_optimizer(args)
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    train_iter = iterators.SerialIterator(train,batchsize)
    valid_iter = iterators.SerialIterator(valid,batchsize,repeat=False,shuffle=False)
    test_iter = iterators.SerialIterator(test,batchsize,repeat=False,shuffle=False)

    max_epoch = 100
    mean_loss=0
    delta = 1e-7

    model = L.Classifier(model)
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
    trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'l1/W/data/std', 'elapsed_time']))
    trainer.extend(extensions.ParameterStatistics(model.predictor.fc6, {'std': np.std}))
    trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std.png'))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()

    test_evaluator = extensions.Evaluator(test_iter, model, device=gpu_id)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])

    from chainer import serializers

    serializers.save_npz('CNNetwork.model', model)
    """
    Networkの構造をグラフ化
    dot -Tpng mnist_result/cg.dot -o mnist_result/cg.png
    Image(filename='mnist_result/cg.png')
    """

            #空白
