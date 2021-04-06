# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:03:32 2021

@author: 13124
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

### 1
path = os.path.join(os.getcwd())
var = pd.read_excel(os.path.join(path,'Variable Description.xlsx'))
df = pd.read_csv(os.path.join(path,'Covid002.csv'), encoding='ISO-8859-1')

var1 = var[(var['Source'] == 'Opportunity Insights') | (var['Source'] == 'PM_COVID')]
var1 = var1['Variable']

df1 = df[df.columns.intersection(var1)]
df2 = df[['county', 'state', 'deathspc']]

covid = pd.concat([df1, df2], axis=1)


### 2
covid.describe()


### 3
covid = covid.dropna()

### 4
covid['state'].nunique()
covid = pd.get_dummies(covid, columns = ['state'])


### 5
X = covid.copy()
X.pop('state_Alabama')
X.pop('county')
y = X.pop('deathspc')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

### 6
# a)
reg = LinearRegression().fit(X_train, y_train)

# R-squared r2_score(y_train, y_pred1)
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))

# MSE
y_pred1 = reg.predict(X_train)
mean_squared_error(y_train, y_pred1)
y_pred2 = reg.predict(X_test)
mean_squared_error(y_test, y_pred2)


### 7
# Ridge
ridge = Ridge(normalize = True)

alpha_param = (10**np.linspace(start=-2, stop=2, num=100))

def vector_values(grid_search, trials):
    mean_vec = np.zeros(trials)
    std_vec = np.zeros(trials)
    i = 0
    final = grid_search.cv_results_
    
    #Using Grid Search's 'cv_results' attribute to get mean and std for each paramter
    for mean_score, std_score in zip(final["mean_test_score"], final["std_test_score"]):
        mean_vec[i] = -mean_score
        # negative sign used with mean.score() to get positive mean squared error
        std_vec[i] = std_score
        i = i+1

    return mean_vec, std_vec

#Creating a parameters grid
param_grid = [{'alpha': alpha_param }]

#Running Grid Search over the alpha (regularization) parameter
grid_search_ridge = GridSearchCV(ridge, param_grid, cv = 10, 
                                 scoring = 'neg_mean_squared_error')
grid_search_ridge.fit(X_train, y_train)

#Calling the vector_values function created to calculate mean and std vectors
mean_vec_r, std_vec_r = vector_values(grid_search_ridge, 100)

plt.figure(figsize=(12,10))
plt.title('Ridge Regression', fontsize= 20)
plt.plot(np.log(alpha_param), mean_vec_r)
plt.errorbar(np.log(alpha_param), mean_vec_r, yerr = std_vec_r)
plt.ylabel("MSE", fontsize= 20)
plt.xlabel("log(Alpha)", fontsize= 20)
plt.show()

# LASSO
lasso = Lasso(normalize = True)

alpha_param2 = (10**np.linspace(start=-3, stop=2, num=100))

#Creating a parameters grid
param_grid2 = [{'alpha': alpha_param2}]

#Running Grid Search over the alpha (regularization) parameter
grid_search_lasso = GridSearchCV(lasso, param_grid2, cv = 10, 
                                 scoring = 'neg_mean_squared_error')
grid_search_lasso.fit(X_train, y_train)

#Calling the vector_values function created to calculate mean and std vectors
mean_vec_l, std_vec_l = vector_values(grid_search_lasso, 100)

plt.figure(figsize=(12,10))
plt.title('Lasso Regression', fontsize= 20)
plt.plot(np.log(alpha_param2), mean_vec_l)
plt.errorbar(np.log(alpha_param2), mean_vec_l, yerr = std_vec_l)
plt.ylabel("MSE", fontsize= 20)
plt.xlabel("log(Alpha)", fontsize= 20)
plt.show()


# Optimal alpha
alpha_param[np.where(mean_vec_r == min(mean_vec_r))][0]
alpha_param2[np.where(mean_vec_l == min(mean_vec_l))][0]

ridgereg = Ridge(alpha = alpha_param[np.where(mean_vec_r == min(mean_vec_r))][0],
                 normalize=True)
ridgereg = ridgereg.fit(X_train, y_train)

lassoreg = Lasso(alpha_param2[np.where(mean_vec_l == min(mean_vec_l))][0], 
                 normalize = True)
lassoreg = lassoreg.fit(X_train, y_train)

### 8
# Find the optimal MSE score
print(min(mean_vec_r))
print(min(mean_vec_l))

# ridge
print(ridgereg.score(X_train, y_train))
print(ridgereg.score(X_test, y_test))

y_pred_r1 = ridgereg.predict(X_train)
mean_squared_error(y_train, y_pred_r1)
y_pred_r2 = ridgereg.predict(X_test)
mean_squared_error(y_test, y_pred_r2)

# lasso
print(lassoreg.score(X_train, y_train))
print(lassoreg.score(X_test, y_test))

y_pred_l1 = lassoreg.predict(X_train)
mean_squared_error(y_train, y_pred_l1)
y_pred_l2 = lassoreg.predict(X_test)
mean_squared_error(y_test, y_pred_l2)
