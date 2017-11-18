#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:52:31 2017

@author: malavikabindhi
"""
from __future__ import division
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
mpl.use('Agg')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import os, itertools, subprocess
import csv

def load_data():
    df = pd.read_csv('oasis_longitudinal.csv')
    raw_data=df[['MR Delay','Age','EDUC','MMSE','CDR','eTIV','nWBV','ASF','M/F','SES','Group']]
    return raw_data

def preprocess_data(raw_data):
    # Dropping 16 and 2 missing data points in 'SES' and 'MMSE', respectively
    raw_data_cleaned=raw_data.dropna()

    # Converting 'M/F' and 'SES' from numerical to categorical input
    xnum = raw_data_cleaned[['MR Delay','Age','EDUC','MMSE','CDR','eTIV','nWBV','ASF']]
    xcat_p = raw_data_cleaned[['M/F','SES']] # M/F (Gender): 0: Female; 1: Male.
    y_p = raw_data_cleaned[['Group']]

    le = preprocessing.LabelEncoder()
    xcat=xcat_p.apply(le.fit_transform)
    x=pd.concat([xcat,xnum],axis=1,join='inner')

    # Converting 'Group' (Demented or Nondemented) from numerical to categorical value
    y=y_p.apply(le.fit_transform) # 0: Converted; 1: Demented; 2: Nondemented
    return x,y


def split_data(x,y):
    train_split=0.7 # fraction of the data used in a training set
    m=x.shape[0] # number of data points

    x_train=x.iloc[0:int(m*train_split),:]
    y_train=y.iloc[0:int(m*train_split),:]
    x_test=x.iloc[int(m*train_split)+1:m-1,:]
    y_test=y.iloc[int(m*train_split)+1:m-1,:]
    return x_train, y_train, x_test, y_test

"""def load_data():
    f = open('oasis_longitudinal.csv')
    csv_f = csv.reader(f)
    y = [];
    train = np.zeros((373,8))
    i= 0
    for row in csv_f:
        if(i > 0):
            print(row)
            if(row[2]=='Demented'):
                y.append(1)
            else:
                y.append(0)
            if(row[5]=='M'):
                train[i-1,2] = 1
            else:
                train[i-1,2] = 0
            if(row[6]=='R'):
                train[i-1,3] = 1
            else:
                train[i-1,3] = 0
        i= i+1
    data = np.genfromtxt('oasis_longitudinal.csv', skip_header=True, delimiter=',')
    train[:,0:2] = data[:,3:5]
    train[:,4:7] = data[:,12:15]
    return train, y"""

def add_intercept(X_):
    #####################
    m = X_.shape[0]
    X = np.ones((m, 4))
    X[:, 1:4] = X_
    ###################
    return X


def dist(a, b):
    dist = 0
    ################
    dist = np.sum((a-b) * (a-b))
    ################
    return dist

def findNNOutput(X,y, Xpredict, ypredict):
    clf = MLPClassifier(hidden_layer_sizes=(5, 2), max_iter=150, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
    #clf = MLPClassifier(solver='sgd', alpha=1e-05,hidden_layer_sizes=(20, 3),random_state=1)
    print(clf.fit(X, y))
    outputs =  clf.predict(Xpredict)
    print(outputs);
    '''df = outputs == y[1:,:]
    count = 0
    k =0
    for i in df:
        if(i):
            count = count + 1
    print(count/len(outputs))'''
    print("Training set score: %f" % clf.score(X, y))
    print("Test set score: %f" % clf.score(Xpredict, ypredict))
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('dementia.png')

def main():
    raw_data=load_data()
    x,y=preprocess_data(raw_data)
    x_train, y_train, x_test, y_test=split_data(x,y)
    '''raw_train, y = load_data()
    "X =  add_intercept(raw_train)"
    X_train, X_test, y_train, y_test = train_test_split(raw_train, y)'''
    print(x_train.shape[0])
    print(x_test.shape[0])
    findNNOutput(x_train,y_train, x_test, y_test)

if __name__ == '__main__':
    main()