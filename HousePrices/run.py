#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:09:40 2021

@author: leonaodole
"""

import sklearn 
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import datasets
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
def Problem2(xvals,yvals, train_x = None, train_y = None, test_x = None, test_y = None):
    if train_x == None:
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
        a = a * 2 * len(train_y)
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
    

def Problem3_4(xvals,yvals, train_x = None, train_y = None, test_x = None, test_y = None):
    if train_x == None:
        train_x = xvals[0:400,]
        test_x = xvals[401:,]
        train_y = yvals[0:400]
        test_y = yvals[401:]
    
    test_a = [0.001,0.01,0.1,1,10,100]
    test_aRidge = [a* 2 * np.shape(train_y)[0] for a in test_a]
    
    Lasso = sklearn.linear_model.LassoCV(fit_intercept=True, cv = 5, alphas=test_a)
  
    
    Lasso.fit(train_x,train_y)
    print("Lasso Optimal alpha: ", Lasso.alpha_)
    pred = Lasso.predict(test_x)
    LassoRMSE = math.sqrt(sklearn.metrics.mean_absolute_error(test_y, pred))
    print("Lasso Testing Error: ",LassoRMSE)
    
    RR = sklearn.linear_model.RidgeCV(fit_intercept = True, cv = 5, alphas = test_aRidge)
    RR.fit(train_x,train_y)
    print("Ridge Regression Optimal alpha: ", RR.alpha_)
    pred = RR.predict(test_x)
    RRrmse = math.sqrt(sklearn.metrics.mean_absolute_error(test_y, pred))
    print("Ridge Regression Testing Error: ", RRrmse)
    
def Problem_5():
    
    with open("./Problem5_Q1.txt", 'w') as file:
        file.write(Problem1(xvals, yvals))
   

#xvals, yvals = readInData("./house_scale.txt") 
#Problem1(xvals, yvals)
#Problem2(xvals, yvals)
#Problem3_4(xvals,yvals)    

#xvals, yvals = readInData("./house.txt")
#Problem3_4(xvals,yvals) 






