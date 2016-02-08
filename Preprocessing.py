__author__ = 'alex'

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from enum import Enum

class Scaler(Enum):
    none = 0
    standard = 1
    minMax = 2
    normalize = 3

def shuffle_data(X,y):
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

def zscore(X):
    return (X - X.mean(axis=1).reshape(len(X),1))/X.std(axis=1).reshape(len(X),1)

def PCA(X):
    pass
    #PCA - leave 185 dimensions

def XY_split(df):
    X = df.drop('IsFailure', axis=1).values
    Y = df['IsFailure'].values.ravel()
    return X,Y

def zeros_ones(df):
    return df[df['IsFailure'] < 10e-12],\
           df[df['IsFailure'] >= 10e-12]

def zero_one_split(df):
    zeros, ones = zeros_ones(df)
    return XY_split(zeros), XY_split(ones)


def scaler_switch(scaler):
    switch = {
        Scaler.none: lambda x: x,
        Scaler.standard: lambda x: StandardScaler().fit_transform(x),
        Scaler.minMax: lambda x: MinMaxScaler().fit_transform(x),
        Scaler.normalize: lambda x: normalize(x, axis=0)
    }
    return switch.get(scaler, lambda x: x)

