#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:06:01 2021

@author: leonaodole
"""

import sklearn
from liblinear.liblinearutil import * 
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
#import requests
import numpy as np
import pandas as pd
from numpy import linalg


def readIndices(filename):
    file = open(filename)
    indices = []
    for line in file:
        indices.append(int(line.strip()))
    return indices
    
    
def readDataSet(filename):
    # url = url
    # file = requests.get(url)
    # temp = open("./temp.txt", 'wt')
    # temp.write(file.text)
    return load_svmlight_file(filename)
    


def chooseC(train_y, train_X,test_y, test_X, c_Candidates,s):
    accuracy = []
    for c in c_Candidates:
        print("C:", c)
        params = "-s "+ str(s) + " -c " + str(c) + " -v 5 -q"
        model = train(train_y, train_X, params)
        accuracy.append(model)
        #params = "-s 0 -c " + str(c) +" -q"
        #predict(test_y, test_X, train(train_y, train_X, params))
    return c_Candidates[np.argmax(accuracy)], accuracy[np.argmax(accuracy)]
    
    

def reportBestModel(train_y, train_X,test_y, test_X, C,s):
    params = "-s " + str(s) + " -c " + str(C) 
    #model = (train_y, train_X, params)
    print("Testing Accuracy with C =", C)
    pred,accr,score = predict(test_y, test_X, train(train_y, train_X, params))
    return pred, accr, score
    

##Problem 1 A

#Read in the Breast Data
print("\n \n Breast Data Logisitc Regression:")
breast_test = readIndices("./breast-cancer-scale-test-indices.txt")
breast_train = readIndices("./breast-cancer-scale-train-indices.txt")

breast_test = [b -1 for b in breast_test]
breast_train = [b -1 for b in breast_train]

breast_dataX, breast_dataY = readDataSet("./breast-cancer_scale.txt")

#Define test/train split
breast_train_y = np.take(breast_dataY, breast_train)
breast_train_X = np.take(breast_dataX.toarray(), breast_train, axis = 0)
breast_test_y = np.take(breast_dataY, breast_test)
breast_test_X = np.take(breast_dataX.toarray(), breast_test, axis = 0)

#Find and test candidate Cs
c_Candidates = [0.1,1,10,100,1000]
bestC, accr = chooseC(breast_train_y, breast_train_X,breast_test_y, breast_test_X, c_Candidates, s=0)

#Report model accuracy on full testing data when using best c and full training set
reportBestModel(breast_train_y, breast_train_X, breast_test_y, breast_test_X, bestC, s=0)

##Problem 1B
print("\n \n Sonar Data Logisitc Regression: ")
#Read in the Sonar Data
sonar_test = readIndices("./sonar-scale-test-indices.txt")
sonar_train = readIndices("./sonar-scale-train-indices.txt")

sonar_test = [s -1 for s in sonar_test]
sonar_train = [s -1 for s in sonar_train]

sonar_dataX, sonar_dataY = readDataSet("./sonar_scale.txt")


#Define the test/train split
sonar_train_y = np.take(sonar_dataY, sonar_train)
sonar_train_X = np.take(sonar_dataX.toarray(), sonar_train, axis =0)
sonar_test_y = np.take(sonar_dataY, sonar_test)
sonar_test_X = np.take(sonar_dataX.toarray(), sonar_test, axis = 0)

#Test Candidate Cs 
bestCsonar, accrSonar = chooseC(sonar_train_y, sonar_train_X, sonar_test_y, sonar_train_X, c_Candidates, s=0)

#Report model accuracy 

### FIX ME ###
reportBestModel(sonar_train_y, sonar_train_X, sonar_test_y, sonar_test_X, bestCsonar,s=0)


#Problem 2A SVM
print("\n \n Breast Data SVM:")
bestC, accr = chooseC(breast_train_y, breast_train_X,breast_test_y, breast_test_X, c_Candidates, s=3)
reportBestModel(breast_train_y, breast_train_X, breast_test_y, breast_test_X, bestC, s=3)


print("\n \n Sonar Data SVM:")
bestCsonar, accrSonar = chooseC(sonar_train_y, sonar_train_X, sonar_test_y, sonar_train_X, c_Candidates, s=3)
reportBestModel(sonar_train_y, sonar_train_X, sonar_test_y, sonar_test_X, bestCsonar,s=3)

#Problem 2B SVM Kernel


#Problem 3


def preprocessing(cov_data):
    #Create New arrays 
    
    cov_data_rescaled = np.empty((cov_data.shape[0], cov_data.shape[1]-1))
    cov_data_standardized = np.empty((cov_data.shape[0], cov_data.shape[1]-1))
    cov_data_normalized = np.empty((cov_data.shape[0], cov_data.shape[1]-1))
    
    
    #Feature scaling procedure:
    #   1.Calculate quantities needed for transformation by feature
    #   2.Calculate the scaled x'
    for col in range(0,54):
        
        #Needed for rescaling
        x_max = np.max(cov_data.iloc[:, col])
        x_min = np.min(cov_data.iloc[:, col])
        diff  = x_max - x_min
        
        #Needed for standardization
        x_mean = np.mean(cov_data.iloc[:, col])
        x_std = np.std(cov_data.iloc[:, col])
        
        #Needed for Normalization
        x_norm = np.linalg.norm(cov_data.iloc[:, col])
        
        for row in range(0, cov_data.shape[0]):
            x = cov_data.iloc[row,col]
            
            #Rescaled Component
            #   x' = (x - x_min/x_max-x_min)
            cov_data_rescaled[row,col] = (x - x_min)/diff
            
            #Standardized Component
            #   x' = (x - x_mean)/x_std
            cov_data_standardized[row,col] = (x -x_mean)/x_std
            
            #Normalization Component
            cov_data_normalized[row,col] = x/x_norm
            
            
            
    return cov_data_rescaled, cov_data_standardized, cov_data_normalized




def splitTestTrain(train_i, test_i, data, labels):
    train_X = np.take(data, train_i, axis=0)
    train_y = np.take(labels, train_i)
    test_X = np.take(data, test_i, axis=0)
    test_y = np.take(labels, test_i)
    return train_X, train_y, test_X, test_y


def metricsOfModel(y_pred, y_true, y_score):
    accr = sklearn.metrics.accuracy_score(y_true, y_pred)
    print("Accuracy:", accr)
    f1= sklearn.metrics.f1_score(y_true, y_pred)
    print("F1 Score:", f1)
    AUC = sklearn.metrics.roc_auc_score(y_true, y_score)
    print("AUC:", AUC)
    



cov_data = pd.read_csv("./covtype.data", header=None)
cov_labels = cov_data.iloc[:,54]
cov_labels = [1 if c ==2 else -1 for c in cov_labels]

cov_train = readIndices("./covtype.train.index.txt")
cov_train = [c-1 for c in cov_train]

cov_test = readIndices ("./covtype.test.index.txt")
cov_test = [c-1 for c in cov_test]
    
cov_scaled, cov_std, cov_norm = preprocessing(cov_data)

cov_data = cov_data.iloc[:,0:54].to_numpy()

c_Candidates = [0.1,1,10,100,1000]
s = 3

#Unaltered chooseC
print("\n \n Covtype Data ")
train_X, train_y, test_X, test_y = splitTestTrain(cov_train, cov_test, cov_data, cov_labels)
C, accr = chooseC(train_y, train_X, test_y, test_X, c_Candidates, s)
print("Best C:",bestC, "\n Accuracy:", accr )
pred, accr, score = reportBestModel(train_y, train_X, test_y, test_X, C, s)
metricsOfModel(pred, test_y,score)
sklearn.metrics.RocCurveDisplay.from_predictions(test_y, pred )
#Scaled chooseC
print("\n \n Covtype Data Scaled")
train_X, train_y, test_X, test_y = splitTestTrain(cov_train, cov_test, cov_scaled, cov_labels)
C, accr = chooseC(train_y, train_X, test_y, test_X, c_Candidates, s)
print("Best C:",bestC, "\n Accuracy:", accr )
pred, accr, score = reportBestModel(train_y, train_X, test_y, test_X, C, s)
metricsOfModel(pred, test_y,score)
sklearn.metrics.RocCurveDisplay.from_predictions(test_y, pred )

#Standardized chooseC
print("\n \n Covtype Data Standardized")
train_X, train_y, test_X, test_y = splitTestTrain(cov_train, cov_test, cov_std, cov_labels)
C, accr = chooseC(train_y, train_X, test_y, test_X, c_Candidates, s)
print("Best C:",bestC, "\n Accuracy:", accr )
pred, accr, score = reportBestModel(train_y, train_X, test_y, test_X, C, s)
metricsOfModel(pred, test_y,score)
sklearn.metrics.RocCurveDisplay.from_predictions(test_y, pred )

#Normalized chooseC
print("\n \n Covtype Data Normalized")
train_X, train_y, test_X, test_y = splitTestTrain(cov_train, cov_test, cov_norm, cov_labels)
C, accr = chooseC(train_y, train_X, test_y, test_X, c_Candidates, s)
print("Best C:",bestC, "\n Accuracy:", accr )
pred, accr, score = reportBestModel(train_y, train_X, test_y, test_X, C, s)
metricsOfModel(pred, test_y,score)
sklearn.metrics.RocCurveDisplay.from_predictions(test_y, pred )












