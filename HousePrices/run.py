#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 09:09:40 2021

@author: leonaodole
"""

import sklearn 
from sklearn import linear_model
import pandas as pd 
import numpy as np

#load data
scaled_df = pd.read_csv("./house_scale.txt", sep =  "\s+" , header=None, dtype=(str))

#Clean Data, Take out the index
for row in range(0,506):
    for col in range(1,14):
        scaled_df.values[row,col] = scaled_df.values[row,col].replace(str(col) +":", "")
        
#Test and Training Split


    
#Problem 1
a =1
Lasso_1 = sklearn.linear_model.Lasso(alpha=a, fit_intercept = True)
Lasso_1.fit(scaled_df[2:13], scaled_df[1]) #ValueError: Found input variables with inconsistent numbers of samples: [11, 506]
RR_1 = sklearn.linear_model.Ridge(alpha=a, fit_intercept = True)