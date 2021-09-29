#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:42:06 2021

@author: leonaodole
"""
import numpy as np
from numpy import linalg
#import pandas as pd
import os

#Read in Doc Files


    


def ReadInFiles(extension,list_bank,docnum = None):
    if docnum == None:
        for filename in os.listdir(extension):
            if filename[0] ==".":
                continue
            else:
                with open(os.path.join(extension, filename)) as file:  
                    f = file.readlines()
                    f = [s.strip("\n") for s in f]
                    list_bank.append(f)
    else:
        i= 0
        for filename in os.listdir(extension):
            if filename[0] ==".":
                continue
            else:
                with open(os.path.join(extension, filename)) as file: 
                    docnum.append(filename)
                    i=i+1
                    f = file.readlines()
                    f = [s.strip("\n") for s in f]
                    list_bank.append(f)    
        
        
        
        
        
wordbank =[]
docnum = []
 
ReadInFiles("./docs/", wordbank, docnum)

#print(wordbank[0])         
            
#Create Dictionary of Unique words            
uniquewords = []
for doc in wordbank:
    for word in doc:
        if word not in uniquewords:
            uniquewords.append(word)
            


#Create dtm for documents 
dtm = np.full((len(uniquewords), len(wordbank)),0)
#rows are the terms
#each col vector is a document

for doc in range(0, len(wordbank)):
    #read each doc
    for word in range(0,len(wordbank[doc])):
        #increment the proper word
        dtm[uniquewords.index(wordbank[doc][word]),doc] = dtm[uniquewords.index(wordbank[doc][word]),doc]  + 1

queries = []

#add queiries to the querie arry as rows 

ReadInFiles("./queries/", queries)

#create quierie   

qmatrx = np.full((len(queries), len(uniquewords)),0, )


for q in range(0, len(queries)):
    for w in range(0, len(queries[q])):
        # print(queries[q][w])
        # print(uniquewords.index(queries[q][w]))
        qmatrx[q, uniquewords.index(queries[q][w])] =  qmatrx[q, uniquewords.index(queries[q][w])] +1
    

#need to dot product   
# print(qmatrx)
similarity = np.matmul(qmatrx,dtm)

dotsimilarity = np.empty((5,500))
for q in range(0,5):
    for d in range(0,500):
        dotsimilarity[q,d] = np.dot(qmatrx[q,:],dtm[:,d])/(np.linalg.norm(dtm[:,d])*np.linalg.norm(qmatrx[q,:]))

print("Using the dot product the 10 query/document pairs with the largest similarity scores are:")
for i in np.argsort(similarity, axis=None)[-10:]:
     a = np.unravel_index(i,similarity.shape)
     print((a[0]+1,docnum[a[1]]))
     print(similarity[a])
     # print(similarity[np.unravel_index(i,similarity.shape)])
    

print("Using the cosine dot product the 10 query/document pairs with the largest similarity scores are:")
for i in np.argsort(dotsimilarity, axis=None)[-10:]:
     a = np.unravel_index(i,dotsimilarity.shape)
     print((a[0]+1,docnum[a[1]]))
     print(dotsimilarity[a])

     # print(similarity[np.unravel_index(i,similarity.shape)])
    



    
 
