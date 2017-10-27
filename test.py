#!/usr/local/bin/python3
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import numpy as np
from sklearn import linear_model


f = open("SUM_wo_noise.csv")
feature_names = f.readline().split()

num_rows = 2000001

shortFile = []
for i in range (num_rows):
    row = f.readline()
    if row == "":
        print ("hit limit:", i)
        break
    shortFile.append(row)
#     print (shortFile[i])


data = np.loadtxt(shortFile, dtype=int, delimiter=";", usecols=range(0,len(feature_names)-1))
clas = np.loadtxt(shortFile, dtype=str, delimiter=";", usecols=len(feature_names)-1)

n = data[:, 0]      # row number    1*p     [all rows, 0th column]
X = data[:, 1:-1]   # features      n*p     [all rows, 1st column to last-1 column]
Y = data[:, -1]     # target        1*p     [all rows, last column]
# print ( clas ) 

# print ("n[0]:", n[0])
# print ("X[0]:", X[0])
# print ("Y[0]:", Y[0])

# print (n)
# print (X)
# print (Y)
num_tests = int(num_rows*0.1) 
print ("num_rows:", num_rows)
print ("num_tests:", num_tests)
x_train = X[:-num_tests]
x_test = X[-num_tests:]
y_train = Y[:-num_tests]
y_test = Y[-num_tests:]

# print ("x_train:\n", x_train)
# print ("x_test:\n", x_test)
# print ("y_train:\n", y_train)
# print ("y_test:\n", y_test)

regr = linear_model.LinearRegression()
fit = regr.fit(x_train, y_train)
print ("fit:", fit)

mean = np.mean((regr.predict(x_test)-y_test)**2)
print ("mean:", mean)

score = regr.score(x_test, y_test)
print ("score:", score)

print ("x_in:", x_test[0])
print ("target:", y_test[0])
print ("prediction:", regr.predict([x_test[0]]))



