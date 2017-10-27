#!/usr/local/bin/python3

def RMSE():
    sq_e = 0
    size = 0
    for predicted, target in zip(predictions, targets):
        if predicted - target != 0:
            sq_e += 1
        size += 1
    return sq_e/size

def R2():
    xy = 0
    x = 0
    x2 = 0
    y = 0
    y2 = 0
    n = 0
    for predicted, target in zip(predicitons, targets):
        x += predicted
        x2 += predicted ** 2
        y += target
        y2 += target ** 2
        xy += predicted * target
        n += 1
    r = ( ( n * xy ) - (x * y) ) / np.sqrt( ( (n * x2) - (x ** 2) ) * ( (n * y2) - (y ** 2) ) )
    r = r ** 2
    return r

def Accuracy():
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

def F1():
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
