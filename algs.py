#!/usr/local/bin/python3
import numpy as np
import tensorflow as tf

class algs ():
    def __init__ (self,name,main_method,train_method,predict_method,alg_type,metrics,framework,feature_columns):
        self.name = name
        self.main_method = main_method
        self.train_method = main_method
        self.predict_method = predict_method
        self.alg_type = alg_type
        self.metrics = metrics
        self.framework = framework
        self.addModel()
        self.addFeatures()

    def addModel(self):
        pass
    def addFeatures(self):
        pass
    def Predict(self):
        pass
    def Train(self):
        pass


class sklAlg (algs):
    def Predict(self, X):
        self.main_method.predict(X)

    def Train(self, X, Y, C):
        if self.alg_type == "REG":
            self.main_method.fit(X, Y)
        else:
            self.main_method.fit(X, C)

class tensorAlg (algs):
    def Predict(self, X):
        predict_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X},
            num_epochs=1,
            shuffle=False)
        self.main_method.predict(input_fn=predict_fn, steps=None)


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
        self.main_method.train(input_fn=train_fn, steps=2000)
    
    def addModel():
        model_dir = tempfile.mkdtemp()
        self.main_method = self.main_method(
            model_dir=model_dir, feature_columns=self.features)

    def addFeatures():
        self.features = [ tf.feature_column.numeric_column(col) for col in feature_columns ] 
