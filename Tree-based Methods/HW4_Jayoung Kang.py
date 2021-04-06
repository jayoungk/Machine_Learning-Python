# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 07:12:08 2021

@author: 13124
"""

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


### Chapter 6
## Question 9
# a) 
path = os.path.join(os.getcwd())
college = pd.read_csv(os.path.join(path,'Data-College.csv'))

college.drop(['Unnamed: 0'], axis=1, inplace=True)
college = pd.get_dummies(college, columns = ['Private'])
college.drop(['Private_No'], axis=1, inplace=True)
X = college.drop(['Apps'], axis=1)
y = college['Apps']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state = 2)


# b) 
reg = LinearRegression().fit(X_train, y_train)
# MSE
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred)


# e) 
# reference: https://www.statology.org/principal-components-regression-in-python/
mse = []
np.random.seed(2)

for i in range(1, 18):
    pcr = make_pipeline(StandardScaler(), PCA(n_components=i), LinearRegression())
    mse.append(-np.mean(cross_val_score(pcr, X_train, y_train, 
                                           scoring='neg_mean_squared_error', cv=17)))

mse = DataFrame(mse,columns=['Value'])
mse.index += 1

plt.plot(mse)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('MSE by number of principle components')

mse[['Value']].idxmin() 

pcr = make_pipeline(StandardScaler(), PCA(n_components=17), LinearRegression())
pcr = pcr.fit(X_train, y_train)
y_predict = pcr.predict(X_test)
mean_squared_error(y_test, y_predict)



# f) 
mse_pls = []
np.random.seed(2)

for i in range(1, 18):
    pls = PLSRegression(n_components=i)
    mse_pls.append(-np.mean(cross_val_score(pls, X_train, y_train, 
                                            scoring='neg_mean_squared_error', cv=17)))

mse_pls = DataFrame(mse_pls,columns=['Value'])
mse_pls.index += 1

plt.plot(mse_pls)
plt.xlabel('Number of Components')
plt.ylabel('MSE')
plt.title('MSE by components (PLS)')

mse_pls[['Value']].idxmin() 

pls = PLSRegression(n_components=14)
pls.fit(X_train, y_train)
y_predict = pls.predict(X_test)
mean_squared_error(y_test, y_predict)


# g) 
# R-squared
print(reg.score(X_test, y_test))
print(pcr.fit(X_train, y_train).score(X_test, y_test))
print(pls.fit(X_train, y_train).score(X_test, y_test))



### Chapter 8
## Question 9
# i) 
oj = pd.read_csv(os.path.join(path,'oj.csv'))
oj.drop(['Unnamed: 0'], axis=1, inplace=True)
oj = pd.get_dummies(oj, columns = ['Store7'])
oj.drop(['Store7_No'], axis=1, inplace=True)

X = oj.drop(['Purchase'], axis=1)
y = oj['Purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=270, random_state = 1)


# ii) 
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)

model_forplot = DecisionTreeClassifier(max_depth=3, random_state=1)
model_forplot.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
print('\nTrain error rate of unpruned tree: ', 1 - accuracy_score(y_train, y_pred_train))

y_pred_forplot = model_forplot.predict(X_train)
cm = confusion_matrix(y_train, y_pred_forplot)
print(cm)
print('\nTrain error rate of unpruned tree: ', 1 - accuracy_score(y_train, y_pred_forplot))



# iii)
plt.figure(figsize=(12,12))
tree.plot_tree(model_forplot, fontsize=10)
plt.show()


# iv)
y_predict = model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print(cm)
print('\nTest error rate: ', 1 - accuracy_score(y_test, y_predict))

y_predict = model_forplot.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print(cm)
print('\nTest error rate: ', 1 - accuracy_score(y_test, y_predict))


# v)
# ref: https://michael-fuchs-python.netlify.app/2019/11/30/introduction-to-decision-trees/
# Determine the optimal tree size by tuning the ccp_alpha argument in 
# scikit-learn’s DecisionTreeClassifier. You can use GridSearchCV for this purpose.
path_model = model.cost_complexity_pruning_path(X_train, y_train) 
ccp_alphas = path_model.ccp_alphas
    
models = []
for ccp_alpha in ccp_alphas:
    mod = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    mod.fit(X_train, y_train)
    models.append(mod)

train_scores = [mod.score(X_train, y_train) for mod in models]
test_scores = [mod.score(X_test, y_test) for mod in models]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show();


model_alpha = DecisionTreeClassifier(random_state=1, ccp_alpha=0.00747714)
tree_size = np.arange(2,20)
parameters = {'max_leaf_nodes': tree_size}
cv_tree = GridSearchCV(model_alpha, parameters)
cv_tree.fit(X_train, y_train)

cv_scores = []
for mean_score in zip(cv_tree.cv_results_["mean_test_score"]):
    cv_scores.append(mean_score[0])


# vi)
# Produce a plot with tree size on the x-axis and cross-validated classification error
# rate on the y-axis calculated using the method in the previous question. Which tree
# size corresponds to the lowest cross-validated classification error rate?
cv_error_rates = cv_scores
for i in range(len(cv_scores)):
    cv_error_rates[i] = 1 - cv_scores[i]

cv_error_rates = DataFrame(cv_error_rates,columns=['Value'])
cv_error_rates.index += 2
cv_error_rates['Value'].min()
cv_error_rates[['Value']].idxmin() 

plt.figure(figsize=(10,8))
sns.lineplot(x=tree_size, y=cv_error_rates['Value'])
plt.xlabel("Tree size", fontsize= 16)
plt.ylabel("CV error rate", fontsize= 16)
plt.title("CV error by tree size", fontsize= 18)


# vii)
# Produce a pruned tree corresponding to the optimal tree size obtained using 
# crossvalidation. If cross-validation does not lead to selection of a pruned tree, then
# create a pruned tree with five terminal nodes.
model_pruned = DecisionTreeClassifier(max_leaf_nodes=8, random_state=1)
model_pruned.fit(X_train, y_train)
plt.figure(figsize=(8,8))
tree.plot_tree(model_pruned, fontsize=10)
plt.show();


# viii)
# Compare the training error rates between the pruned and unpruned trees. 
# Which is higher?
y_pred_train = model.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)
print(cm)
print('\nTrain error rate of unpruned tree: ', 1 - accuracy_score(y_train, y_pred_train))

y_pred_train_p = model_pruned.predict(X_train)
cm2 = confusion_matrix(y_train, y_pred_train_p)
print(cm2)
print('\nTrain error rate of pruned tree: ', 1 - accuracy_score(y_train, y_pred_train_p))

# xi)
# Compare the test error rates between the pruned and unpruned trees. 
# Which is higher?
y_pred_test = model.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred_test)
print(cm3)
print('\nTest error rate of unpruned tree: ', 1 - accuracy_score(y_test, y_pred_test))

y_pred_test_p = model_pruned.predict(X_test)
cm4 = confusion_matrix(y_test, y_pred_test_p)
print(cm4)
print('\nTest error rate of pruned tree: ', 1 - accuracy_score(y_test, y_pred_test_p))
