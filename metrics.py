#!/usr/local/bin/python3
import numpy as np

def RMSE(predictions, targets):
    sq_e = 0
    size = 0
    total = 0
    minT = min(targets)
    maxT = max(targets)
    MinT = None
    MaxT = None
    for predicted, target in zip(predictions, targets):
        sq_e += (abs(target - predicted) ** 2)
        print ("taget:", target)
        print ("predicted:", predicted)
        print ("cur sq_e:", sq_e)
        if MinT == None or MinT > target :
            MinT = target
        if MaxT == None or MaxT < target :
            MaxT = target
        # if abs(predicted - target) < 0.0001:
        #     sq_e += 1
        size += 1

    if size == 0:
        print ("size = 0")
        return 0
    elif (maxT - minT) == 0:
        print ("dif = 0")
        return 1
    else:
        print ("size:", size)
        print ("sq_e:", sq_e)
        print ("sqrt:", np.sqrt(sq_e/size))
        print ("max -min :", maxT - minT)
        print ("Max,  Min :", MaxT, MinT)
        print ("Max - Min :", MaxT - MinT)
        return np.sqrt(sq_e/size)/(maxT - minT)

def R2(predictions, targets):
    xy = 0
    x = 0
    x2 = 0
    y = 0
    y2 = 0
    n = 0
    for predicted, target in zip(predictions, targets):
        x += predicted
        x2 += predicted ** 2
        y += target
        y2 += target ** 2
        xy += predicted * target
        n += 1
    dom = np.sqrt( abs( ( (n * x2) - (x ** 2) ) * ( (n * y2) - (y ** 2) ) ) )
    if dom == 0:
        r = 0
    else:
        r = ( ( n * xy ) - (x * y) ) / dom 
    r = r ** 2
    return r

def Accuracy(predictions, targets):
    correct = 0
    size = 0
    for predicted, target in zip(predictions, targets):
        if predicted - target == 0:
            correct += 1
        size += 1
    if size != 0:
        return correct/size
    else:
        return 0

def F1(predictions, targets):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for predicted, target in zip(predictions, targets):
        if predicted == 1 and target == 0:
            FP += 1
        elif predicted == 1 and target == 1:
            TP += 1
        elif predicted == 0 and target == 0:
            TN += 1
        elif predicted == 0 and target == 1:
            FN += 1
        else:
            print ("Error calculating f1")

    if TP != 0:
        precision = TP / (TP + FP)
        recall = TP / ( TP + FN )
    else:
        precision = 0
        recall = 0
         
    if precision != 0 or recall != 0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    else:
        f1 = 0
    return f1
