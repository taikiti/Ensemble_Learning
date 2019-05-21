import chainer
import numpy as np
from chainer import Variable,Chain
import chainer.links as L
import chainer.functions as F

class CNNetwork(Chain):#2 3
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,50, ksize=5, stride=3)
            self.conv2=L.Convolution2D(None, 50, ksize=5, stride=3)
            self.fc3=L.Linear(None, 250)
            self.fc4=L.Linear(None, 50)
            self.fc5=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)
        return y
class CNNetwork1(Chain):#2 3
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,100, ksize=5, stride=1)
            self.conv2=L.Convolution2D(None, 100, ksize=5, stride=1)
            self.fc3=L.Linear(None, 500)
            self.fc4=L.Linear(None, 50)
            self.fc5=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)
        return y

class CNNetwork2(Chain):#1 3
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,50, ksize=5, stride=1)
            self.fc3=L.Linear(None, 100)
            self.fc4=L.Linear(None, 50)
            self.fc5=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)
        return y
class CNNetwork3(Chain):#1 3
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,100, ksize=5, stride=1)
            self.fc3=L.Linear(None, 2500)
            self.fc4=L.Linear(None, 50)
            self.fc5=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)
        return y

class CNNetwork4(Chain): #3 3
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,50, ksize=5, stride=1)
            self.conv2=L.Convolution2D(None, 50, ksize=5, stride=1)
            self.conv3=L.Convolution2D(None, 50, ksize=5, stride=1)
            self.fc3=L.Linear(None, 500)
            self.fc4=L.Linear(None, 50)
            self.fc5=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv3(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)
        return y
class CNNetwork5(Chain): #3 3
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,100, ksize=5, stride=1)
            self.conv2=L.Convolution2D(None, 100, ksize=5, stride=1)
            self.conv3=L.Convolution2D(None, 100, ksize=5, stride=1)
            self.fc3=L.Linear(None, 500)
            self.fc4=L.Linear(None, 50)
            self.fc5=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv3(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)
        return y

class CNNetwork6(Chain):#1 2
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,50, ksize=5, stride=1)
            self.fc3=L.Linear(None, 100)
            self.fc4=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        y = self.fc4(h)
        return y
class CNNetwork7(Chain):#1 2
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,100, ksize=5, stride=1)
            self.fc3=L.Linear(None, 150)
            self.fc4=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        y = self.fc4(h)
        return y

class CNNetwork8(Chain):#2 2
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,50, ksize=5, stride=1)
            self.conv2=L.Convolution2D(None, 50, ksize=5, stride=1)
            self.fc3=L.Linear(None, 250)
            self.fc4=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        y = self.fc4(h)
        return y
class CNNetwork9(Chain):#2 2
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(None,100, ksize=5, stride=1)
            self.conv2=L.Convolution2D(None,100, ksize=5, stride=1)
            self.fc3=L.Linear(None, 500)
            self.fc4=L.Linear(None, 14)

    def __call__(self, x):
        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        y = self.fc4(h)
        return y
