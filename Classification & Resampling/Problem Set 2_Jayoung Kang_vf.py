# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:00:37 2021

@author: 13124
"""

import math
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

### Chapter 4
## Question 6
# a)
math.exp(-0.5) / (1+math.exp(-0.5))

# b)
np.log(1)
2.5/0.05


## Question 7
x=4
mu=10
sigma=6
pi=3.14

yes = 0.8*(1/(sigma*np.sqrt(2*math.pi)))*math.exp(-0.5*((x-mu)/sigma)**2)
no = 0.2*(1/(sigma*np.sqrt(2*math.pi)))*math.exp(-0.5*((x)/sigma)**2)

yes/(yes+no)


## Question 9
# a)
0.37 / 1.37
# b)
0.16/(1-0.16)


## Question 11
# a)
path = os.path.join(os.getcwd())
auto = pd.read_csv(os.path.join(path,'Data-Auto.csv'))
med = np.median(auto['mpg'])
auto['mpg01'] = np.where(auto['mpg'] > med, 1, 0)

# b)
sns.pairplot(auto, 
             y_vars=['mpg01'],
             x_vars=['mpg','cylinders','displacement','horsepower']);

sns.pairplot(auto, 
             y_vars=['mpg01'],
             x_vars=['weight','acceleration','year','origin']);

col = list(auto.columns) 
col.remove('mpg01')
col.remove('name')
col.remove('Unnamed: 0')

for i in col:
    sns.boxplot(x='mpg01', y=i, data=auto)
    plt.figure();

fig, axes = plt.subplots(1, 4, figsize=(20,5))
sns.boxplot(ax=axes[0], x='mpg01', y='horsepower', data=auto);
sns.boxplot(ax=axes[1], x='mpg01', y='acceleration', data=auto);
sns.boxplot(ax=axes[2], x='mpg01', y='weight', data=auto);
sns.boxplot(ax=axes[3], x='mpg01', y='displacement', data=auto);

# c)
# X = auto.drop(['mpg01','name','Unnamed: 0'], axis=1)
X = auto[['horsepower', 'acceleration', 'weight', 'displacement']]
Y = auto['mpg01']
X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size=0.30, 
                                                    random_state=207)

# d)
lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, Y_train)
y_pred_lda = lda_model.predict(X_test)
print((1-accuracy_score(Y_test, y_pred_lda))*100)
print(np.mean(y_pred_lda != Y_test)*100)


# e)
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train, Y_train)
y_pred_qda = qda_model.predict(X_test)
print((1-accuracy_score(Y_test, y_pred_qda))*100)
print(np.mean(y_pred_qda != Y_test)*100)

# f)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred_logreg = logreg.predict(X_test)
print((1-accuracy_score(Y_test, y_pred_logreg))*100)
print(np.mean(y_pred_logreg != Y_test)*100)



### Chapter 5
## Question 5
# a)
default = pd.read_csv(os.path.join(path,'Data-Default.csv'))

X = default[['income', 'balance']]
y = default['default'].astype('category').cat.codes
logreg.fit(X, y)


# b)~c)
for i, seed in enumerate([12, 20, 208]):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, 
                                                      random_state=seed)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_posterior = logreg.predict_proba(X_val)
    y_predicted = np.where(y_posterior[:,1] > 0.5, 1, 0)
    print(np.mean(y_predicted != y_val)*100)

# c)
default['student'] = default['student'].astype('category').cat.codes

X = default[['income', 'balance', 'student']]
y = default['default'].astype('category').cat.codes

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, 
                                                  random_state=22)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_posterior = logreg.predict_proba(X_val)
y_predicted = np.where(y_posterior[:,1] > 0.5, 1, 0)
print(np.mean(y_predicted != y_val)*100)
