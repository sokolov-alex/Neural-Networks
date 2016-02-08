__author__ = 'alex'

import numpy as np
from matplotlib import pyplot as plt

def add_bias(x):
    ones = np.ones((x.shape[0],1))
    return np.hstack((x, ones))

def plot_digit(x, y):
    if x.shape[0]>784:
        x = x[:-1]
    x = np.reshape(x, (28,28))
    plt.imshow(x, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y)
    plt.show()

def load_mnist(path='Data/mnist.npz', bias = True):
    data = np.load(path)
    X_train = data['images_train']/255
    y_train = data['labels_train']
    X_test = data['images_test']/255
    y_test = data['labels_test']
    if bias:
        X_train = add_bias(X_train)
        X_test = add_bias(X_test)
    return X_train, y_train, X_test, y_test