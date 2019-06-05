import chainer
import numpy as np

import matplotlib.pyplot as plt
from chainer import cuda,optimizers
from chainer import Variable,Chain,dataset, datasets
from chainer import serializers
import pickle
import chainer.links as L
from chainer.cuda import to_cpu

from convolution import CNNetwork
from convolution import CNNetwork1
from convolution import CNNetwork2
from convolution import CNNetwork3
from convolution import CNNetwork4
from convolution import CNNetwork5
from convolution import CNNetwork6
from convolution import CNNetwork7
from convolution import CNNetwork8
from convolution import CNNetwork9


model0 = L.Classifier(CNNetwork())
serializers.load_npz('./models/CNNetwork0.model', model0)
model1 = L.Classifier(CNNetwork1())
serializers.load_npz('./models/CNNetwork1.model', model1)
model2 = L.Classifier(CNNetwork2())
serializers.load_npz('./models/CNNetwork2.model', model2)
model3 = L.Classifier(CNNetwork3())
serializers.load_npz('./models/CNNetwork3.model', model3)
model4 = L.Classifier(CNNetwork4())
serializers.load_npz('./models/CNNetwork4.model', model4)
model5 = L.Classifier(CNNetwork5())
serializers.load_npz('./models/CNNetwork5.model', model5)
model6 = L.Classifier(CNNetwork6())
serializers.load_npz('./models/CNNetwork6.model', model6)
model7 = L.Classifier(CNNetwork7())
serializers.load_npz('./models/CNNetwork7.model', model7)
model8 = L.Classifier(CNNetwork8())
serializers.load_npz('./models/CNNetwork8.model', model8)
model9 = L.Classifier(CNNetwork9())
serializers.load_npz('./models/CNNetwork9.model', model9)

gpu_id = -1  # CPUで計算をしたい場合は、-1を指定してください

if gpu_id >= 0:
    model0.to_gpu(gpu_id)
    model1.to_gpu(gpu_id)
    model2.to_gpu(gpu_id)
    model3.to_gpu(gpu_id)
    model4.to_gpu(gpu_id)
    model5.to_gpu(gpu_id)
    model6.to_gpu(gpu_id)
    model7.to_gpu(gpu_id)
    model8.to_gpu(gpu_id)
    model9.to_gpu(gpu_id)

test_pickle=pickle.load(open('test.dat','rb'),encoding = 'bytes')
test_data = test_pickle['data']
test_label = test_pickle['target']

test =  datasets.TupleDataset(test_data, test_label)

count = 0
error_num = []
for i in range(len(test)):
    x,t = test[i]

    x0 = model0.xp.asarray(x)
    x1 = model1.xp.asarray(x)
    x2 = model2.xp.asarray(x)
    x3 = model3.xp.asarray(x)
    x4 = model4.xp.asarray(x)
    x5 = model5.xp.asarray(x)
    x6 = model6.xp.asarray(x)
    x7 = model7.xp.asarray(x)
    x8 = model8.xp.asarray(x)
    x9 = model9.xp.asarray(x)


    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
         y0 = model0.predictor(x0[None,...])
         y1 = model1.predictor(x1[None,...])
         y2 = model2.predictor(x2[None,...])
         y3 = model3.predictor(x3[None,...])
         y4 = model4.predictor(x4[None,...])
         y5 = model5.predictor(x5[None,...])
         y6 = model6.predictor(x6[None,...])
         y7 = model7.predictor(x7[None,...])
         y8 = model8.predictor(x8[None,...])
         y9 = model9.predictor(x9[None,...])

    y0 = to_cpu(y0.array)
    y1 = to_cpu(y1.array)
    y2 = to_cpu(y2.array)
    y3 = to_cpu(y3.array)
    y4 = to_cpu(y4.array)
    y5 = to_cpu(y5.array)
    y6 = to_cpu(y6.array)
    y7 = to_cpu(y7.array)
    y8 = to_cpu(y8.array)
    y9 = to_cpu(y9.array)


    pred_label = []
    pred_label.append(y0.argmax(axis=1)[0])
    pred_label.append(y1.argmax(axis=1)[0])
    pred_label.append(y2.argmax(axis=1)[0])
    pred_label.append(y3.argmax(axis=1)[0])
    pred_label.append(y4.argmax(axis=1)[0])
    pred_label.append(y5.argmax(axis=1)[0])
    pred_label.append(y6.argmax(axis=1)[0])
    pred_label.append(y7.argmax(axis=1)[0])
    pred_label.append(y8.argmax(axis=1)[0])
    pred_label.append(y9.argmax(axis=1)[0])

    chk = 0
    for k in range(10):
        print('ネットワークの予測'+str(k)+':', pred_label[k])
        if pred_label[k] == t:
            chk+=1
    print('答え:',t)
    if chk>7:
        print('正解')
        count+=1
    else:
        print('不正解')
        error_num.append(i)
    print('正答率:',count/len(test))
print('count:',count)
