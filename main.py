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

from algs import *
# from algs import sklAlg
# from algs import tensorAlg
# from algs import loadAlgs
# from algs import allMetrics
# from algs import alg_types
from metrics import *
from printer import printResults

def readlines (filename, **kwargs):
    global upperLimit
    global limit
    f = open(filename)
    feature_names = np.array(f.readline().replace(',', ';').replace(' ', '_').split(";"))
    print (feature_names)
    shortFile = []
    i = 0
    while True:
        row = f.readline().replace(',', ';')
        limit = i
        if row == "" or (i >= upperLimit and upperLimit > 0):
            print ("hit limit:", i)
            break
        shortFile.append(row)
        i += 1
    if shortFile:
        X, Y, Classes, name2num = parsedata(shortFile, len(feature_names))
        return X, Y, Classes, feature_names
    else:
        return None, None, None, None

def parsedata (shortFile, length):
    data = np.loadtxt(shortFile, dtype=float, delimiter=";", usecols=range(1,length-1))
    clas = np.loadtxt(shortFile, dtype=str, delimiter=";", usecols=length-1)
#     n = data[:, 0]      # row number    1*p     [all rows, 0th column]
    X = data[:, 0:-1]   # features      n*p     [all rows, 1st column to last-1 column]
    Y = data[:, -1]     # target        1*p     [all rows, last column]
    
    class_name2num = {}
    # class_num2name = {}
    classes = np.unique(clas)
    if "Large Number" in classes:
        classes[0], classes[2] = classes[2], classes[0]
    classes_len = len(classes)
    for i, className in enumerate(classes):
        # class_name2num[className] = i 
        if i <= (classes_len / 2): 
            class_name2num[className] = 0
            # class_num2name[0] = className
        else:
            class_name2num[className] = 1
            # class_num2name[1] = className

    Classes = np.array([ class_name2num[name] for name in clas ])
    # print (Classes)

    return X, Y, Classes, class_name2num #, class_num2name

def loadDatasets(DataSets):
    for FileName in DataSets:
        # DataSets[FileName] = readlines(FileName)
        X, Y, C, F = readlines(FileName)
        DataSets[FileName]["X"] = X
        DataSets[FileName]["Y"] = Y
        DataSets[FileName]["C"] = C
        DataSets[FileName]["F"] = F

def trainDatasets(datasets):
    dataset_results = []
    for dset in datasets:
        dataset_results.append(trainAlgs(datasets[dset]))
    return dataset_results

def trainAlgs(dset):
    global algs
    alg_results = []
    for alg in algs:
        alg_results.append(trainChunks(alg, dset))
    return alg_results

def trainChunks(alg, dset):
    global chunks
    global limitBroken 
    global chunk
    chunk_results = [] 
    for chunk in chunks:
        if not limitBroken:
            if chunk > limit:
                print ( "Limit broke" )
                limitBroken = True
                chunk_results.append(trainFolds(alg, dset, limit))
            else:
                chunk_results.append(trainFolds(alg, dset, chunk))
        else:
            chunk_results.append((1, None))
    limitBroken = False
    return chunk_results

def trainFolds(alg, dset, chunk):
    global folds
    fold_results = []
    kf = KFold(n_splits=folds)
    for index in kf.split(dset["X"][0:chunk]):
        train_index = index[0]
        test_index = index[1]
        fold_results.append(assesFold(alg, dset, chunk, train_index, test_index)) 
    avg = averageFolds(fold_results, alg.metrics)
    return avg
def assesFold(alg, dset, chunk, train_index, test_index):
    X, Y, C = getTrainTest(dset, chunk, train_index, test_index)
    alg.addFeatures(dset["F"])
    alg.Train(X["train"], Y["train"], C["train"])
    # print (alg.Predict(X["test"]))
    return alg.test(alg.Predict(X["test"]), Y["test"], C["test"])

def averageFolds(fold_results, metrics):
    results = []
    for metric in metrics:
        results.append(0)
    for fold in fold_results:
        for i, metric_value in enumerate(fold):
            results[i] += metric_value
    avg_results = np.array([ result / len(fold_results) for result in results ])
    return avg_results


def getTrainTest(dset, chunk, train_index, test_index): 
    X_train, X_test = dset["X"][0:chunk][train_index], dset["X"][0:chunk][test_index]
    Y_train, Y_test = dset["Y"][0:chunk][train_index], dset["Y"][0:chunk][test_index]
    C_train, C_test = dset["C"][0:chunk][train_index], dset["C"][0:chunk][test_index]
    return { "train":X_train, "test":X_test }, { "train":Y_train, "test":Y_test }, { "train":C_train, "test":C_test }

def trainSplit(alg, dset, chunk):
    X_train, X_test = dset["X"][0:(chunk*70)], dset["X"][(chunk*70):]
    Y_train, Y_test = dset["Y"][0:(chunk*70)], dset["Y"][(chunk*70):]
    C_train, C_test = dset["C"][0:(chunk*70)], dset["C"][(chunk*70):]


datasets = {
        "SUM_wo_noise.csv" : {},
        # "SUM_w_noise.csv" : {},
        # "housing_prices.csv" : {},
        # "kc_housing_prices.csv": {}
        }
dataset_names = [
        "The SUM Dataset(without noise)",
        # "The SUM Dataset(with noise)",
        # "Housing Prices Dataset",
        # "KC Housing Prices Dataset"
        ]
folds = 10
upperLimit = 1000
limit = upperLimit
limitBroken = False
chunks = [100, 500]
# chunks = [100, 500, 1000, 5000, 10000, 50000, 100000]
loadDatasets(datasets)
algs = loadAlgs()
# print (datasets)
# for alg in algs:
#     print (alg.name)
results = trainDatasets(datasets)
# print (results)
printerResults = printResults(results, allMetrics, alg_types, dataset_names, alg_metrics_names_lists, alg_names)
# for line in printerResults:
    # print (printerResults[line])
import csv
with open("output.csv", 'w') as f:
    f.write(",")
    for chunk in chunks:
        f.write("{},".format(chunk))
    f.write("\n")
    for i in range(len(printerResults)):
        for chunk in printerResults[i]:
            f.write("{},".format(chunk))
        f.write("\n")
    # wr = csv.writer(f)
    # for i in range(len(printerResults)):
    #     wr.writerow(printerResults[i])





