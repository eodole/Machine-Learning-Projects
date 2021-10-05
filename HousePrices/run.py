#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:09:40 2021

@author: leonaodole
"""

import sklearn 
from sklearn import linear_model
from sklearn import metrics
import math
import matplotlib 
from matplotlib import pyplot as plt

import pandas as pd 
import numpy as np

#load data
def readInData(file_name):
    scaled_df = pd.read_csv(file_name, sep =  "\s+" , header=None, dtype=(str))
    
    #Clean Data, Take out the index
    for row in range(0,506):
        for col in range(1,14):
            scaled_df.values[row,col] = scaled_df.values[row,col].replace(str(col) +":", "")
            #scaled_df.values[row,col] = float(scaled_df.values[row,col])
    #Test and Training Split
    yvals = scaled_df[0]
    xvals = np.array(scaled_df.loc[:,1:13])
    return(xvals, yvals)

xvals, yvals = readInData("./house_scale.txt")   

#Problem 1
def Problem1(xvals,yvals): 
    a =1
    Lasso_1 = sklearn.linear_model.Lasso(alpha=a, fit_intercept = True)
    Lasso_1.fit(xvals, yvals)
    print("Optimal Lasso Parameters:")
    print(Lasso_1.coef_)#print #ValueError: Found input variables with inconsistent numbers of samples: [11, 506]
    RR_1 = sklearn.linear_model.Ridge(alpha=a, fit_intercept = True)
    RR_1.fit(xvals,yvals)
    print("Optimal Ridge Regression Parameters:")
    print(RR_1.coef_)

#Problem 2
def Problem2(xvals,yvals):
    train_x = xvals[0:400,]
    test_x = xvals[401:,]
    train_y = yvals[0:400]
    test_y = yvals[401:]
    
    #Fit the Lasso
    test_a = [0,0.001,0.01,0.1,1,10,100]
    LassoRMSEtest = []
    LassoRMSEtrain = []
    RidgeRMSEtest = []
    RidgeRMSEtrain = []
    for a in test_a:
        #Train the Lasso Model
        Lasso_2 = sklearn.linear_model.Lasso(alpha =a, fit_intercept = True )
        Lasso_2.fit(train_x,train_y)
        
        #Predict and append RMSE
        pred = Lasso_2.predict(test_x)
        LassoRMSEtest.append(math.sqrt(sklearn.metrics.mean_absolute_error(test_y, pred)))
        pred = Lasso_2.predict(train_x)
        LassoRMSEtrain.append(math.sqrt(sklearn.metrics.mean_absolute_error(train_y, pred)))
        
        #Train the Ridge Regression Model 
        RR_2 = sklearn.linear_model.Ridge(alpha =a, fit_intercept = True)
        RR_2.fit(train_x,train_y)
        pred = RR_2.predict(test_x)
        RidgeRMSEtest.append(math.sqrt(sklearn.metrics.mean_absolute_error(test_y, pred)))
        pred =RR_2.predict(train_x)
        RidgeRMSEtrain.append(math.sqrt(sklearn.metrics.mean_absolute_error(train_y, pred)))
    
    #Create plot of RMSE
    values = range(len(test_a))
    
    plt.plot(values, LassoRMSEtest, "b--")
    plt.plot(values, LassoRMSEtrain, "g-")
    plt.plot(values, RidgeRMSEtest, "r--")
    plt.plot(values, RidgeRMSEtrain, "y--")   
    plt.xticks(values,test_a)
    plt.xlabel("Alpha")
    plt.ylabel("Testing Error")
    plt.legend(["Lasso Test", "Lasso Train", "Ridge Test", "Ridge Train"], loc = "upper left")
    plt.show()
    

def Problem3(xvals,yvals):
    print("")
    
    


#Problem1(xvals, yvals)
#Problem2(xvals, yvals)
    





