__author__ = 'alex'

import numpy as np
from matplotlib import pyplot as plt
import sklearn
import sklearn.linear_model
import pandas as pd
import h5py
import copy
import os
from subprocess import call

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_cifar(datapath = 'Data', url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'):
    if not os.path.exists("{}/cifar-100-python.tar.gz".format(datapath)):
        call("wget {}".format('http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'), shell=True)
        print("Downloaded cifar dataset.\n")

    cifar_path = os.path.abspath("{}/cifar-100".format(datapath))
    if not os.path.exists(cifar_path):
        call(
            "tar -zxvf cifar-100-python.tar.gz && mv cifar-100-python cifar-100",
            shell=True
        )
        print("Extracted to {}.".format(cifar_path))

    def load_cifar(train_file):
        data = []
        labels = []

        d = unpickle(os.path.join(cifar_path, train_file))

        data = d['data']
        coarse_labels = d['coarse_labels']
        fine_labels = d['fine_labels']
        length = len(d['fine_labels'])

        return data.reshape(length, 3, 32, 32), np.array(coarse_labels), np.array(fine_labels)

    X, y_c, y_f = load_cifar("train")

    Xt, yt_c, yt_f = load_cifar("test")


    if not os.path.exists(os.path.join(cifar_path, 'train.h5')):
        train_filename = os.path.join(cifar_path, 'train.h5')
        test_filename = os.path.join(cifar_path, 'test.h5')

        comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
        # Train
        with h5py.File(train_filename, 'w') as f:
            f.create_dataset('data', data=X, **comp_kwargs)
            f.create_dataset('label_coarse', data=y_c.astype(np.int_), **comp_kwargs)
            f.create_dataset('label_fine', data=y_f.astype(np.int_), **comp_kwargs)
        with open(os.path.join(cifar_path, 'train.txt'), 'w') as f:
            f.write(train_filename + '\n')
        # Test
        with h5py.File(test_filename, 'w') as f:
            f.create_dataset('data', data=Xt, **comp_kwargs)
            f.create_dataset('label_coarse', data=yt_c.astype(np.int_), **comp_kwargs)
            f.create_dataset('label_fine', data=yt_f.astype(np.int_), **comp_kwargs)
        with open(os.path.join(cifar_path, 'test.txt'), 'w') as f:
            f.write(test_filename + '\n')

        print('Converted to "{}".\n'.format(cifar_path))

    return X, y_c, y_f, \
           Xt, yt_c, yt_f


class mnist():
    def __init__(self):
        pass

    def add_bias(self, x):
        ones = np.ones((x.shape[0],1))
        return np.hstack((x, ones))

    def plot_digit(self, x, y):
        if x.shape[0]>784:
            x = x[:-1]
        x = np.reshape(x, (28,28))
        plt.imshow(x, cmap='Greys',  interpolation='nearest')
        plt.title('true label: %d' % y)
        plt.show()

    def load_mnist(self, path='Data/mnist.npz', bias = True):
        data = np.load(path)
        X_train = data['images_train']/255
        y_train = data['labels_train']
        X_test = data['images_test']/255
        y_test = data['labels_test']
        if bias:
            X_train = self.add_bias(X_train)
            X_test = self.add_bias(X_test)
        return X_train, y_train, X_test, y_test


'''
def load_cifar_pickle():
    train = unpickle('Data/cifar-100/train')
    test = unpickle('Data/cifar-100/test')
    meta = unpickle('Data/cifar-100/meta')
    return train, test, meta

def convert_to_hdf5():
    def to_hdf5(data, name):
        pd.DataFrame(data).to_hdf('Data/cifar-100/{0}.hdf5'.format(name), name, mode = 'w')

    to_hdf5(train['data'], 'X_train')
    to_hdf5(test['data'], 'X_test')

    to_hdf5(train['fine_labels'], 'y_train')
    to_hdf5(test['fine_labels'], 'y_test')

def load_cifar_hdf5(path = 'Data/cifar-100'):
    X_train = pd.read_hdf(path+'/X_train.hdf5')
    y_train = pd.read_hdf(path+'/y_train.hdf5')
    X_test = pd.read_hdf(path+'/X_test.hdf5')
    y_test = pd.read_hdf(path+'/y_test.hdf5')
    return X_train, y_train, X_test, y_test

train, test, meta = load_cifar_pickle()

convert_to_hdf5()

X_train, y_train, X_test, y_test = load_cifar_hdf5()

print 'pickle object keys: ', train.keys()

print 'train shape: ', X_train.shape
'''