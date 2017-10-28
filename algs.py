#!/usr/local/bin/python3
import numpy as np
import tensorflow as tf
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

# from algs import sklAlg
# from algs import tensorAlg
# from algs import loadAlgs
from metrics import *

class algs ():
    def __init__ (self,name,main_method,alg_type,metrics,framework):
        self.name = name
        self.main_method = main_method
        self.model = self.main_method
        self.alg_type = alg_type
        self.metrics = metrics
        self.framework = framework
        self.addModel()

    def test(self, predicted, Y, C):
        if self.alg_type == "REG":
            target = Y
        else:
            target = C
        metric_results = []
        for metric in self.metrics:
            metric_results.append(metric(predicted, target))
        return metric_results

    def addModel(self):
        pass
    def addFeatures(self, feature_columns):
        pass
    def Predict(self, X):
        print ("in base p")
        pass
    def Train(self, X, Y, C):
        print ("in base t")
        pass


class sklAlg (algs):
    def Predict(self, X):
        return self.model.predict(X)

    def Train(self, X, Y, C):
        if self.alg_type == "Regression":
            fitRes = self.model.fit(X, Y)
        else:
            self.model.fit(X, C)

class tensorAlg (algs):
    def Predict(self, X):
        predict_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            num_epochs=1,
            shuffle=False)
        self.model.predict(input_fn=predict_fn, steps=None)

    def Train(self, X, Y, C):
        if self.alg_type == "REG":
            target = Y
        else:
            target = C
        train_fn =  tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            y=target,
            num_epochs=10,
            shuffle=False)
        self.model.train(input_fn=train_fn, steps=2000)
    
    def addFeatures(self, feature_columns):
        features = [ tf.feature_column.numeric_column(col) for col in feature_columns ] 
        model_dir = tempfile.mkdtemp()
        self.model = self.main_method(
            model_dir=model_dir, feature_columns=features)


alg_methods = [ 
    linear_model.LinearRegression(),
    linear_model.Ridge (alpha = .5), 
    LogisticRegression(),
    KNeighborsClassifier() 
    ] 
alg_names = [
    "LinearRegression", 
    "LinearRidge",
    "LogisticRegression",
    "KNeighborsClassifier"
    ]
alg_types = [
    "Regression",
    "Regression",
    "Classification",
    "Classification"
    ]
alg_frameworks = [
    "skl",
    "skl",
    "skl",
    "skl"
    ]
alg_metrics_lists = [
    [RMSE, R2],
    [RMSE, R2],
    [Accuracy, F1],
    [Accuracy, F1]
    ]

alg_metrics_names_lists = [
    ["RMSE", "R2"],
    ["RMSE", "R2"],
    ["Accuracy", "F1"],
    ["Accuracy", "F1"]
    ]

allMetrics =[]
for alg in alg_metrics_lists:
    for metric in alg:
        allMetrics.append(metric)


def loadAlgs():
    global alg_methods, alg_names, alg_types, alg_fameworks, alg_metric_lists
    newAlgs = []
    for method, name, _type, framework, metric_list in zip(alg_methods, alg_names, alg_types, alg_frameworks, alg_metrics_lists):
        if framework == "skl":
            newAlgs.append(sklAlg(name, method, _type, metric_list, framework))
        elif framework == "tensor":
            newAlgs.append(tensorAlg(name, method, _type, metric_list, framework))
        elif framework == "???":
            newAlgs.append(sklAlg(name, method, _type, metric_list, framework))
        else:
            print("huge issue in the fucking algs")
    return newAlgs

# algs = loadAlgs()

# for alg in algs:
#     print (alg.name)
#     print (alg.main_method)
#     print (alg.alg_type)
#     print (alg.metrics)
#     print (alg.model)
