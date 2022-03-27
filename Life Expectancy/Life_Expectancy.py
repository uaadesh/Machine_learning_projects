# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 19:20:29 2021

@author: 91638
"""
# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Life_Expectancy_Data.csv')
dataset = dataset.replace(np.nan, 0)
dataset = dataset.drop(columns = ['Country'])
X = pd.get_dummies(dataset)
X = X.iloc[:,[0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]].values
y = dataset.iloc[:,[2]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Using Multiple linear regression model on the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
