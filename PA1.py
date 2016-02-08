# -*- coding: utf-8 -*-
from __future__ import division
import os

__author__ = 'alex'

import numpy as np
import pandas as pd
import math
from time import clock
from  Preprocessing import shuffle_data, zscore
from DataLoader import load_mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import expit as sigmoid
from scipy.optimize import fmin_l_bfgs_b
from matplotlib import pyplot as plt
from numba import jit
#%matplotlib inline


X_train, y_train, X_test, y_test = load_mnist()

X_train = zscore(X_train)
X_test = zscore(X_test)

X_train, y_train = shuffle_data(X_train, y_train)

n_classes = len(set(y_train))


def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()

class LogisticRegression():
    def __init__(self, dim):
        self.w = np.zeros(dim)
        self.eta = 10e-7

    def predict(self, x):
        return sigmoid(x.dot(self.w))

    def cost(self, y_true, y_pred):
        return -(y_true * np.ma.log(y_pred).data + (1-y_true) * np.ma.log(1-y_pred).data).sum()

    def gradient(self, y_true, y_pred, X):
        return -X * (y_true - y_pred).reshape(y_true.shape[0],1)

    def SGD(self, y_true, y_pred, X):
        self.w = self.w - self.eta * self.gradient(y_true,y_pred,X).sum(axis=0)

    def fit(self, X_train, y_train, mini_batch = 100, verbose = 10):
        old_loss = np.inf
        of = 0
        for i in range(30):
            X_train, y_train = shuffle_data(X_train, y_train)
            for b in range(mini_batch, X_train.shape[0]+1, mini_batch):
                x_batch, y_batch = X_train[b-mini_batch:b], y_train[b-mini_batch:b]
                y_pred = self.predict(x_batch)
                self.SGD(y_batch, y_pred, x_batch)

            if i % verbose == 0:
                loss = self.cost(y_batch, y_pred)
                if old_loss - loss >= 0:
                    of -= 1
                    self.eta *= 2
                else:
                    of += 1
                    self.eta /= 3
                    if of > 5:
                        self.eta /= 20
                        if of > 10:
                            break
                #print 'rate: ', self.eta
                old_loss = loss

                #print 'loss: ', loss
            if loss < 0.001:
                #print 'enough (loss = 0.01)'
                break

    def score(self, X, y_true):
        y_pred = (self.predict(X) >= 0.5).astype(int)
        return accuracy_score(y_true, y_pred)


class LogRegBFGS():
    def __init__(self, dim):
        self.w = np.zeros(dim)
        self.eta = 10e-7

    def predict(self, x):
        return sigmoid(x.dot(self.w))

    def cost(self, w, X, y_true):
        y_pred = self.predict(X)
        return -(y_true * np.ma.log(y_pred).data + (1-y_true) * np.ma.log(1-y_pred).data).sum()

    def gradient(self, w, X, y_true):
        y_pred = self.predict(X)
        return (-X * (y_true - y_pred).reshape(y_true.shape[0],1)).sum(axis=0)*10e-3

    def fit(self, X_train, y_train, verbose = 10):
        self.w, loss, g = fmin_l_bfgs_b(self.cost, self.w, fprime=self.gradient, args=(X_train, y_train),callback=None)
        #print 'loss: ', self.cost(self.w, X_train[::500], y_train[::500])

    def score(self, X, y_true):
        y_pred = (self.predict(X) >= 0.5).astype(int)
        return accuracy_score(y_true, y_pred)


class SoftmaxRegression(LogisticRegression):
    def __init__(self, dim):
        LogisticRegression.__init__(self, dim)

    def predict(self, x):
        return np.exp(x.dot(self.w))/np.exp(x.dot(self.w)).sum(axis=1).reshape((x.shape[0],1))

    def cost(self, y_true, y_pred):
        return -(y_true * np.ma.log(y_pred).data).sum().sum()

    def SGD(self, y_true, y_pred, X):
        self.w = self.w - self.eta * self.gradient(y_true,y_pred,X).sum(axis=1).T

    def gradient(self, y_true, y_pred, X):
        return -X.reshape((1, X.shape[0], X.shape[1])).repeat(10,0) * (y_true - y_pred).T.reshape(y_true.shape[1],y_true.shape[0],1)

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred.argmax(axis=1))

@jit
def run_lr():
    dim = X_train.shape[1]

    for i in range(10):
        lr = LogisticRegression(dim)
        y_true = (y_train == i).astype(int)

        lr.fit(X_train, y_true)

        y_true_test = (y_test == i).astype(int)
        #print 'test accuracy: ', lr.score(X_test, y_true_test)

from sklearn.preprocessing import LabelBinarizer
@jit
def run_sr():
    dim = (X_train.shape[1], n_classes)
    lb = LabelBinarizer()
    y_true = lb.fit_transform(y_train)

    sr = SoftmaxRegression(dim)
    sr.fit(X_train, y_true, verbose=1)
    #print 'test accuracy: ', sr.score(X_test, y_test)

def run_lr_bfgs(X_train, y_train):
    dim = X_train.shape[1]
    for i in range(10):
        lr = LogRegBFGS(dim)
        X_train, y_train = shuffle_data(X_train, y_train)
        y_true = (y_train == i).astype(int)

        lr.fit(X_train, y_true)

        y_true_test = (y_test == i).astype(int)
        print('test accuracy: ', lr.score(X_test, y_true_test))
#os.environ['PATH'] = os.environ['PATH']+'/home/alex/anaconda/lib:' TODO: try uninstalling anaconda numpy and scipy and reintalling mkl
#os.environ['LD_LIBRARY_PATH'] = ''




t1 = clock()
run_lr_bfgs(X_train, y_train)
#print 'total time: ', clock()-t1