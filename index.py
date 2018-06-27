# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 08:24:36 2018

@author: Nikhil
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib as plt  


#importing datasets
dataset = pd.read_csv("optical_interconnection_network.csv",";")
dataset = dataset.iloc[0:,0:10]



#average data values 

#processor utilizaiton
a = dataset.iloc[0:,5]  
b = a.str.split(",")
c = []
for i in range(0,640):
    c.append((int(b.iloc[0:][i][1])+int(b.iloc[0:][i][0]))/2)
dataset.drop('Processor_Utilization ',axis = 1, inplace = True)
dataset['Processor_Utilization'] = c


#channel waiting time
a = dataset.iloc[0:,5]  
b = a.str.split(",")
c = []
for i in range(0,640):
    c.append((int(b.iloc[0:][i][1])+int(b.iloc[0:][i][0]))/2)
dataset.drop('Channel_Waiting_Time',axis = 1, inplace = True)
dataset['Channel_Waiting_Time'] = c

#Input_Waiting_Time
a = dataset.iloc[0:,5]  
b = a.str.split(",")
c = []
for i in range(0,640):
    c.append((int(b.iloc[0:][i][1])+int(b.iloc[0:][i][0]))/2)
dataset.drop('Input_Waiting_Time',axis = 1, inplace = True)
dataset['Input_Waiting_Time'] = c


#Network_Response_Time
a = dataset.iloc[0:,5]  
b = a.str.split(",")
c = []
for i in range(0,640):
    c.append((int(b.iloc[0:][i][1])+int(b.iloc[0:][i][0]))/2)
dataset.drop('Network_Response_Time',axis = 1, inplace = True)
dataset['Network_Response_Time'] = c


#Channel_Utilization
a = dataset.iloc[0:,5]  
b = a.str.split(",")
c = []
for i in range(0,640):
    c.append((int(b.iloc[0:][i][1])+int(b.iloc[0:][i][0]))/2)
dataset.drop('Channel_Utilization',axis = 1, inplace = True)
dataset['Channel_Utilization'] = c


#T/R
a = dataset.iloc[0:,4]  
b = a.str.split(",")
c = []
for i in range(0,640):
    if int(b.iloc[0:][i][0]) == 0:
        c.append((int(b.iloc[0:][i][1])+int(b.iloc[0:][i][0]))/2)
    else:
        c.append(int(b.iloc[0:][i][0]))
dataset.drop('T/R',axis = 1, inplace = True)
dataset['T/R'] = c


#splitting data set into x and y
X = dataset.iloc[0:,0:8]
X['T/R'] = dataset.iloc[0:,9]
X = X.values
y = dataset.iloc[0:,8].values



# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 4:10])
X[:, 4:10] = imputer.transform(X[:, 4:10])



#encoding categorical data 
k = []

for i in range(0,640):
    if X[i:i+1,3][0] == 'Client-Server':
        k.append(0)
    else:
        k.append(1)

X[:,3] = k

#encoding categorical data
k = []
l = []
m = []
n = []

for i in range(0,640):
    if X[i:i+1,2][0] == 'UN':
        k.append(1)
    else :
        k.append(0)

for i in range(0,640):
    if X[i:i+1,2][0] == 'HR':
        l.append(1)
    else :
        l.append(0)

for i in range(0,640):
    if X[i:i+1,2][0] == 'BR':
        m.append(1)
    else :
        m.append(0)

for i in range(0,640):
    if X[i:i+1,2][0] == 'PS':
        n.append(1)
    else :
        n.append(0)

#X = np.insert(X,0,k,axis=1)
X[:,2] = k
X = np.insert(X,0,l,axis=1)
X = np.insert(X,0,m,axis=1)
X = np.insert(X,0,n,axis=1)


#splitting the data set into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#svr 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting results
y_pred = regressor.predict(X_test)    

    

