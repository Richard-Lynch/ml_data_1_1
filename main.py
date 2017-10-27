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
        return X, Y, Classes
    else:
        return None, None, None

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
        X, Y, C = readlines(fileName)
        datasets[filename]["X"] = X
        datasets[filename]["Y"] = Y
        datasets[filename]["C"] = C

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
DataSets = {
        "SUM_wo_noise.csv" : {}
        "SUM_w_noise.csv" : {}
        }
# for set in datasets: datasets["set"].append(X), .append(Y), .append(classes)
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
        # alg.train_method(alg.main_method, alg.X)
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
    X, Y = getTrainTest(dset, chunk, train_index, test_index, alg.alg_type)
    alg.train_method(X["train"], Y["train"])
    for metric in alg.metrics_methods:
        metric_results.append(metric(X["train"], Y["train"], X["test"], Y["test"]))
    return metric_results

def getTrainTest(dset, chunk, train_index, test_index, aType): 
    X_train, X_test = dset["X"][range(chunk)][train_index], dset["X"][range(chunk)][test_index]
    if aType == "Regression":
        Y_train, Y_test = dset["Y"][range(cunk)][train_index], dset["Y"][range(chunk)][test_index]
    else:
        Y_train, Y_test = dset["C"][range(chunk)][train_index], dset["C"][test_index]
    return { "train":X_train, "test":X_test }, { "train":Y_train, "test":Y_test }

def trainSKL(alg, X, Y):
    alg.train_method(alg.main_method, X, Y)

def trainTensor(alg, X, Y):

    alg.main_method.train(input_fn=alg.train_method, steps=2000)

def trainTanor(alg, X, Y):
    pass

def testSKL(metric, X, Y, alg):
    return metric(alg.main_method, X, Y)

def testTensor(metric, X, Y, alg):
    pass

def testTanor(metric, X, Y, alg):
    pass


class algs ():
    def __init__ (self,name,main_method,train_method,predict_method,alg_type,metrics,framework,feature_columns):
        self.name = name
        self.main_method = main_method
        self.train_method = main_method
        self.predict_method = predict_method
        self.alg_type = alg_type
        self.metrics = metrics
        self.framework = framework
        if self.framework == "tensor":
            addTensorFeatures()
            addMainModel()
            self.train_method = self.tensorTrain
            self.predict_method = self.tensorPredict
    
    def addTensorModel():
        model_dir = tempfile.mkdtemp()
        self.main_method = self.main_method(
            model_dir=model_dir, feature_columns=self.features)

    def addTensorFeatures():
        self.features = [ tf.feature_column.numeric_column(col) for col in feature_columns ] 
    
    def tensorPredict(self, X, Y):
        predict_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            y=Y,
            num_epochs=10,
            shuffle=False)
        self.main_method.predict(input_fn=predict_fn, steps=None)

    def tensorTrain(self, X, Y):
        train_fn =  tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            y=Y,
            num_epochs=10,
            shuffle=False)
        self.main_method.train(input_fn=train_fn, steps=2000)




upperLimit = 5000000
limit = 0
limit_broken = False
feature_names, shortFile = readlines ("SUM_wo_noise.csv")
X, Y = parsedata (shortFile)
linear_training_time, linear_rmse, linear_r2 = trainLinear(X, Y)
# linear_training_time, linear_rmse, linear_r2 = trainLinear(X, Y)


