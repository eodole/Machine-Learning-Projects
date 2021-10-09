#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:57:02 2021

@author: leonaodole


"""
#Required Packages 
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import math
import numpy as np


    
def readInData(filename): 
    dataset = pd.read_csv(filename, header = None)
    return dataset
    
    
def kNN_Classifier():
    #Load Heart Data
    
    #Build Classifier
   
    #Train on trainSet.txt 
        #Use leave one out cv with k = {1,2,...,10}
    return

#Load Heart Data
train_data = readInData("./heart_trainSet.txt")
train_labels = readInData("./heart_trainLabels.txt")
train_labels = train_labels.values.ravel()
kNN = KNeighborsClassifier(n_neighbors=5)

#Create Cross Validation sets 
cv_splits = model_selection.KFold(n_splits = 5)
i=0
for i_train, i_test in cv_splits.split(train_data,train_labels):
    print("Train:", i_train, "Test:", i_test,i)
    scores = model_selection.cross_val_score(kNN, train_data, train_labels, scoring='accuracy' ,cv= cv_splits)
    print('Accuracy: %.3f (%.3f)' % (scores[i], scores[i]))
    i=i+1