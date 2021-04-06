# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:38:26 2021

@author: 13124
"""

import os

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
import statsmodels.formula.api as smf


# Exercise 2.1. Finish cleaning the datasets
path = os.path.join(os.getcwd())
vote_df = pd.read_csv(os.path.join(path,'vote.csv'))
work_df = pd.read_csv(os.path.join(path,'work.csv'))

# Map labels to 0 and 1
work_mapper = {'flexible': 1, "not flexible": 0}
vote_mapper = {'vote': 1, "did not vote": 0}

work_df['work'] = work_df['work'].replace(work_mapper)
vote_df['vote'] = vote_df['vote'].replace(vote_mapper)

# Convert it to dummies
work_df = pd.get_dummies(work_df, columns=['prcitshp'])
vote_df = pd.get_dummies(vote_df, columns=['prcitshp']) 

# Add column of zeros to vote_df since it has one less category in citizenship
vote_df["prcitshp_FOREIGN BORN, NOT A CITIZEN OF"] = 0

# Check if both are now same
print(len(work_df.columns))
print(len(vote_df.columns))

work_df = pd.get_dummies(work_df, columns=['ptdtrace'])
vote_df = pd.get_dummies(vote_df, columns=['ptdtrace'])

print(len(work_df.columns))
print(len(vote_df.columns))

vote_df['4 or 5 Races'] = 0
work_df['W-B-AI'] = 0
work_df['W-A-HP'] = 0
work_df['Black-Asian'] = 0

# For remaining columns
work_df = pd.get_dummies(work_df, columns=['pesex'])
vote_df = pd.get_dummies(vote_df, columns=['pesex'])
work_df = pd.get_dummies(work_df, columns=['pehspnon'])
vote_df = pd.get_dummies(vote_df, columns=['pehspnon'])
work_df = pd.get_dummies(work_df, columns=['peeduca'])
vote_df = pd.get_dummies(vote_df, columns=['peeduca'])
print(len(work_df.columns))
print(len(vote_df.columns))



# Exercise 2.2
X = work_df.drop('work', axis = 1)
y = work_df['work']

## Define the set of parameters to form combinations in grid search
parameters = {'kernel':('linear', 'sigmoid', 'poly'), 
              'C':[0.1, 1, 10]}

# Run each combination using GridSearch CV
# By specifying cv = 5, GridSearchCV will run 5-fold CV on each combination
svc = SVC(random_state=1)
clf = GridSearchCV(svc, parameters, cv = 5)
clf.fit(X, y)

# Print out the mean_score for each combination
# GridSearchCV uses accuracy scores by default
for params, mean_score, rank in zip(clf.cv_results_["params"], clf.cv_results_["mean_test_score"], clf.cv_results_["rank_test_score"]):
    print(params, mean_score, rank)
    

# Exercise 2.3
clf = SVC(C=0.1, kernel='poly', random_state=1)
clf.fit(X, y)
clf.score(X, y)

from sklearn.metrics import accuracy_score
y_predict = clf.predict(X)
accuracy_score(y, y_predict)


# Exercise 2.4
vote_X = vote_df.drop('vote', axis = 1)

impute_work = clf.predict(vote_X)
accuracy_score(work_df['work'], impute_work)

    

# Exercise 2.5
result = smf.ols('vote ~ impute_work + prtage + np.power(prtage, 2) + pesex_FEMALE', data = vote_df).fit()
print(result.summary())

work_vote_relationship = result.params[1]
work_vote_relationship
#0.3355

# Exercise 2.6
def compute_M(a,b):
    return 1 / (1 - 2 * b) * (1 - (1 - b) * b / a - (1 - b) * b / (1 - a))

a = sum(impute_work)/(impute_work.size)
print(a)

b = 1- 0.8656
print(b)

M = compute_M(a,b)
print(M)

work_vote_bias_correction = work_vote_relationship / M
print(work_vote_bias_correction)
