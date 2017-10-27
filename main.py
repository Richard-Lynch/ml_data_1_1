#!/usr/local/bin/python3
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import numpy as np
from math import sqrt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time
import tensorflow as tf

from functools import wraps

from algs import sklAlg
from algs import tensorAlg
from metrics import *

def readlines (filename, **kwargs):
    upperLimit = 1000
    limit = upperLimit
    f = open(filename)
    feature_names = f.readline().split()
    shortFile = []
    i = 0
    while True:
        row = f.readline()
        if row == "" or i >= upperLimit:
            print ("hit limit:", i)
            limit = i
            break
        shortFile.append(row)
        i += 1
    if shortFile:
        X, Y, Classes, name2num = parsedata(shortFile)
        return X, Y, Classes, feature_names
    else:
        return None, None, None

def parsedata (shortFile, **kwargs):
    data = np.loadtxt(shortFile, dtype=int, delimiter=";", usecols=range(1,len(feature_names)-1))
    clas = np.loadtxt(shortFile, dtype=str, delimiter=";", usecols=len(feature_names)-1)
#     n = data[:, 0]      # row number    1*p     [all rows, 0th column]
    X = data[:, 0:-1]   # features      n*p     [all rows, 1st column to last-1 column]
    Y = data[:, -1]     # target        1*p     [all rows, last column]
    
    class_name2num = {}
    # class_num2name = {}
    classes = np.unique(clas)
    classes_len = len(classes)
    for i, className in enumerate(classes):
        if i < (classes_len / 2): 
            class_name2num[className] = 0
            # class_num2name[0] = className
        else:
            class_name2num[className] = 1
            # class_num2name[1] = className

    Classes = [ class_name2num[name] for name in clas ] 

    return X, Y, Classes, class_name2num #, class_num2name

def loadDatasets(datasets):
    for fileName in datasets:
        X, Y, C, F = readlines(fileName)
        datasets[filename]["X"] = X
        datasets[filename]["Y"] = Y
        datasets[filename]["C"] = C
        datasets[filename]["F"] = F

DataSets = {
        "SUM_wo_noise.csv" : {}
        "SUM_w_noise.csv" : {}
        }

# chunks = [10000, 50000]
chunks = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

loadDatasets(datasets)
results = trainDatasets(datasets)
def trainDatasets(datasets):
    dataset_results = []
    for dset in datasets:
        dataset_results.append(trainAlgs(dset))
    return dataset_results

def trainAlgs(dset):
    global algs
    alg_results = []
    for alg in algs:
        alg_results.append(trainChunks(alg, dset))
    return alg_results

def trainChunks(alg, dset):
    global chunks
    chunk_results = [] 
    for chunk in chunks:
        chunk_results.append(trainFolds(alg, dset, chunk))
    return chunk_results

def trainFolds(alg, dset, chunk):
    global folds
    fold_results = []
    kf = KFold(n_split=folds)
    for i, index in enumerate(kf.split(dset["X"])):
        train_index = index[0]
        test_index = index[1]
       fold_results.append(assesFold(alg, dset, chunk, train_index, test_index)) 
    return averageFolds(fold_results)
        
def assesFold(alg, dset, chunk, train_index, test_index):
    global something
    metric_results = []
    X, Y, C = getTrainTest(dset, chunk, train_index, test_index)
    alg.addFeatures(dset["features"])
    alg.Train(X["train"], Y["train"], C["train"])
    for metric in alg.metrics_methods:
        metric_results.append(metric(alg, X["test"], Y["test"], C["test"]))
    return metric_results

def getTrainTest(dset, chunk, train_index, test_index): 
    X_train, X_test = dset["X"][range(chunk)][train_index], dset["X"][range(chunk)][test_index]
    Y_train, Y_test = dset["Y"][range(cunk)][train_index], dset["Y"][range(chunk)][test_index]
    C_train, C_test = dset["C"][range(chunk)][train_index], dset["C"][test_index]
    return { "train":X_train, "test":X_test }, { "train":Y_train, "test":Y_test }, { "train":C_train, "test":C_test }

