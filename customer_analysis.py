# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 01:32:48 2023

@author: tamer
"""

#1. libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2. data preprocessing
#2.1 data loading
data = pd.read_csv("Churn_Modelling.csv")

x = data.iloc[:,3:13].values
y = data.iloc[:,13].values

#encoder: Categorical -> Numeric
from sklearn import preprocessing

le1 = preprocessing.LabelEncoder()
x[:,1] = le1.fit_transform(x[:,1])

le2 = preprocessing.LabelEncoder()
x[:,2] = le1.fit_transform(x[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough")

x = ohe.fit_transform(x)
x = x[:,1:]

#3. splitting data for training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#4. scaling of data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#5. artificial neural network
import keras

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
classifier.add(Dense(6, kernel_initializer = "uniform", activation = "relu"))

classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))

classifier.compile(optimizer = "adam" ,loss = "binary_crossentropy",metrics = ["accuracy"])

classifier.fit(x_train, y_train, epochs = 50)
y_pred = classifier.predict(x_test)

y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)






