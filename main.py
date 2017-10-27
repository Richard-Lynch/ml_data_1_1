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

from functools import wraps


def trainChunk(alg, X, Y, **kwargs):
    folds = 10
    
    cvs = -cross_val_score(alg, X, Y, scoring='neg_mean_squared_error', cv=folds)
    rmse = [sqrt(mse) for mse in cvs]
    avgrmse = np.sum(rmse) / len(rmse)
    
    r2 = cross_val_score(alg, X, Y, scoring='r2', cv=folds)
    avgr2 = sum(r2) / len(r2)


    # kf = KFold(n_splits=folds)
    # for i, index in enumerate(kf.split(X)):
    #     train_index = index[0]
    #     test_index = index[1]
# #         print("Train:", train_index, "\nTest:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     Y_train, Y_test = Y[train_index], Y[test_index]

    #     fitSet(alg, X_train, Y_train)
    #     mean_sq_e += meanSqE(alg, X_test, Y_test)
    return avgrmse, avgr2

def trainChunks(alg, X, Y, **kwarg):
    global chunks
    global limit_broken
    global chunks_processed
    global limit
    chunk_training_time = {}
    chunk_rmse = []
    chunk_r2 = []
    for i, chunk in enumerate(chunks):
        if chunk > limit:
            if not limit_broken:
                limit_broken = True
                chunk = limit
        chunks_processed.append(chunk)
        print ("ragen", range(chunk))
        rmse, r2 = trainChunk(alg, X[range(chunk)], Y[range(chunk)], log_name=str(chunk), log_time=chunk_training_time)
        chunk_rmse.append(rmse)
        chunk_r2.append(r2)
    return chunk_training_time, chunk_rmse, chunk_r2

def trainLinear(X, Y):
    global linear
    global linear_names
    global chunks
    global chunks_processed
    global limit
    global limit_broken
    alg_training_time = []
    alg_rmse = []
    alg_r2 = []
    for alg, algname in zip(linear, linear_names):
        limit_broken = False
        chunks_processed = []
        print (algname) 
        chunk_training_time, chunk_mse, chunk_r2 = trainChunks(alg, X, Y, log_time=alg_training_time)
        for chunk in chunks:
            chunk_training_time[str(chunk)] = chunk

        for i in range(len(chunks_processed)):
            print (chunk_training_time)
            # if str(chunks_processed[i]) in chunk_training_time:
            print (algname)
            print ("chunk: {}\nmse: {}\nr2: {}\ntime: {}\n".format(chunks_processed[i], chunk_mse[i], chunk_r2[i], chunk_training_time[str(chunks_processed[i])]))
            # else:
                # print ("FUCCCCCKKKKKKKKKKK")
    return alg_training_time, alg_rmse, alg_r2 

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
        X, Y, Classes, name2num, num2name = parsedata(shortFile)
        return feature_names, X, Y, Classes
    else:
        return None, None, None, None

def parsedata (shortFile, **kwargs):
    data = np.loadtxt(shortFile, dtype=int, delimiter=";", usecols=range(1,len(feature_names)-1))
    clas = np.loadtxt(shortFile, dtype=str, delimiter=";", usecols=len(feature_names)-1)
#     n = data[:, 0]      # row number    1*p     [all rows, 0th column]
    X = data[:, 0:-1]   # features      n*p     [all rows, 1st column to last-1 column]
    Y = data[:, -1]     # target        1*p     [all rows, last column]
    
    class_name2num = {}
    class_num2name = {}
    for i, className in enumerate(np.unique(clas)):
        class_name2num[className] = i
        class_num2name[i] = className

    Classes = [ class_map[name] for name in clas ] 

    return X, Y, Classes, class_name2num, class_num2name

def loadDatasets(datasets):
    for fileName in datasets:
        readlines(fileName)
# chunks = [10000, 50000]
chunks = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

alg_methods = [ 
    linear_model.LinearRegression(),
    linear_model.Ridge (alpha = .5)
    LogisticRegression(),
    KNeighborsClassifier() 
    ] 
alg_names = [
    "LinearRegression", 
    "LinearRidge",
    "LogisticRegression",
    "KNeighborsClassifier"
    ]
alg_train_methods = [
    linear_model.LinearRegression.fit,
    linear_model.Ridge.fit,
    LogisticRegression.fit,
    KNeighborsClassifier.fit 
    ]
alg_predict_methods = [
    linear_model.LinearRegression.predict,
    linear_model.Ridge.predict,
    LogisticRegression.predict,
    KNeighborsClassifier.predict
    ]
alg_type = [
    "Regression",
    "Regression",
    "Classification",
    "Classification"
    ]
metrics_methods = {
    "Regression" : {
        "RMSE" : rmse,
        "R2" : r2
        },
    "Classificaion" : {
        "Accuracy" : AC,
        "R2?" : r2
        }
    }
Data_X = [
   [ X ],
   [ X ],
   [ X ],
   [ X ]
   ]
Data_Y = [
    [ Y ],
    [ Y ],
    [ classes ],
    [ classes ]
    ]

DataSets = {
        "SUM_wo_noise.csv" : []
        "SUM_w_noise.csv" : []
        }
# for set in datasets: datasets["set"].append(X), .append(Y), .append(classes)

upperLimit = 5000000
limit = 0
limit_broken = False
feature_names, shortFile = readlines ("SUM_wo_noise.csv")
X, Y = parsedata (shortFile)
linear_training_time, linear_rmse, linear_r2 = trainLinear(X, Y)
# linear_training_time, linear_rmse, linear_r2 = trainLinear(X, Y)

