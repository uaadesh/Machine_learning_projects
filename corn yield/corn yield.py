# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:21:26 2021

@author: 91638
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('corn yield.csv')
dataset = dataset.drop(columns = ['Program','Week Ending','Geo Level','Ag District','Ag District Code','County','County ANSI','Zip Code','Region','watershed_code','Watershed','Commodity','Domain','Domain Category','CV (%)','State ANSI'])
dataset['Value'] = dataset['Value'].str.replace(',', '').astype(float)
X = pd.get_dummies(dataset)
Y = X['Value'].values
X = X.drop(columns = ['Value','Period_YEAR','State_ALABAMA','Data Item_CORN, GRAIN - ACRES HARVESTED'])
X = X.values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

