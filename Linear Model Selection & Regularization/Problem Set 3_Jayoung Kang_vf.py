# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:35:15 2021

@author: namho
"""

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
from IPython.display import display, HTML
import itertools
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

### Chapter 5
## Question 6
# a)
path = os.path.join(os.getcwd())
default = pd.read_csv(os.path.join(path,'Data-Default.csv'), index_col=0)

encoding_dict = {'Yes': 1, 'No': 0}
default['default']=default['default'].map(encoding_dict)

X = default[['income', 'balance']]
X = sm.add_constant(X)
y = default['default']
results = sm.Logit(y,X).fit()
print(results.summary())
print(results.bse)

# b)
# to get random indices of given size
np.random.seed(1)
def get_indices(data,num_samples):
    positive_data = data[data['default'] == 1]
    negative_data = data[data['default'] == 0]
    
    positive_indices = np.random.choice(positive_data.index, int(num_samples / 4), replace=True)
    negative_indices = np.random.choice(negative_data.index, int(30*num_samples / 4), replace=True)
    total = np.concatenate([positive_indices,negative_indices])
    np.random.shuffle(total)
    return total

# similar to boot.fn
def boot_fn(data,index):
    X = data[['balance','income']].loc[index]
    y = data['default'].loc[index]
    
    lr = LogisticRegression()
    lr.fit(X,y)
    intercept = lr.intercept_
    coef_balance = lr.coef_[0][0]
    coef_income = lr.coef_[0][1]
    return [intercept,coef_balance,coef_income]

intercept, coef_balance, coef_income = boot_fn(default, get_indices(default, 100))
print(f'Intercept is {intercept}, the coeff of balance is {coef_balance}, the coeff for income is {coef_income}')

# c)
# bootstrap standard errors
# similar to boot.fn
np.random.seed(1)
def boot(data,func,R):
    intercept = []
    coeff_balance = []
    coeff_income = []
    for i in range(R):
        
        [inter,balance,income] = func(data,get_indices(data,100))
        intercept.append(float(inter))
        coeff_balance.append(balance)
        coeff_income.append(income)
        
    intercept_statistics = {'estimated_value':np.mean(intercept),'std_error':np.std(intercept)}   
    balance_statistics = {'estimated_value':np.mean(coeff_balance),'std_error':np.std(coeff_balance)}
    income_statistics = {'estimated_value':np.mean(coeff_income),'std_error':np.std(coeff_income)}
    return {'intercept':intercept_statistics,'balance_statistices':balance_statistics,'income_statistics':income_statistics}

results = boot(default,boot_fn,1000)

print('Balance - ',results['balance_statistices'])
print('Income - ', results['income_statistics'])


## Question 8
# a)
np.random.seed(1)
y = np.random.normal(size = 100)
X = np.random.normal(size = 100)
y = X - 2*(X**2) + np.random.normal(size=100)
# n is 100, the number of examples and p is 2, the number of features (X and X^2)
# the equation for the model is y = x - 2x^2

# b)
sns.scatterplot(X,y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
# parabolic shape, quadratic, not perfect because of randomness
# positive corr before zero, neg corr after zero

# c)
np.random.seed(1)
for i in range(1, 5):
    poly = PolynomialFeatures(i, include_bias=False)
    predictors = poly.fit_transform(X.reshape(-1,1))
    
    lr = LinearRegression()
    error = -1*cross_val_score(lr, predictors, y, cv=len(X), 
                               scoring='neg_mean_squared_error').mean()
    print(f'For model {i}, error is {error}')

# LOOCV error is really high when the model is just a regular linear regression
# as soon as we add the squared term the error goes down
# but then it increases slightly as we add higher powers to the model - could be due to overfitting
# Best error value if for degree = 2, and since its simulated data, we know that the real relationship is also quadratic

# d)
np.random.seed(5)
for i in range(1, 5):
    poly = PolynomialFeatures(i, include_bias=False)
    predictors = poly.fit_transform(X.reshape(-1,1))
    
    lr = LinearRegression()
    error = -1*cross_val_score(lr, predictors, y, cv=len(X), 
                               scoring='neg_mean_squared_error').mean()
    print(f'For model {i}, error is {error}')
    
# The results we got are absolutely the same, this is because there is no random sampling in LOOCV.
# Everytime we fit n models such athat each time model will be trained on n-1 observations
# and then tested on a left out observation 

# e) model 2 has the smallest LOOCV error

# f)
for i in range(1, 5):
    poly = PolynomialFeatures(i)
    predictors = poly.fit_transform(X.reshape(-1,1))
    
    results = sm.OLS(y, predictors).fit()
    print(results.summary())
# value of adjusted R_squared doesn't increase after quadratic
# only x1 and x2 are sig


### Chapter 6
## Question 11
# a) best subset, forward stepwise, & backwards stepwise selection
# Present and discuss results for the approaches that you consider
# reference: https://github.com/a-martyn/ISL-python/blob/master/Notebooks/ch6_linear_model_selection_and_regularisation_labs.ipynb
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
X = data.drop('CRIM',axis = 1)
y = data['CRIM']

### Best Subset ###
def get_models(k, X, y):
    """
    Fit all possible models that contain exactly k predictors.
    """
    # List all available predictors
    X_combos = itertools.combinations(list(X.columns), k)
    
    # Fit all models accumulating Residual Sum of Squares (RSS)
    models = []
    for X_label in X_combos:
        # Parse patsy formula
        X_smf = ' + '.join(X_label)
        f     = 'CRIM ~ {}'.format(X_smf)
        # Fit model
        model = smf.ols(formula=f, data=pd.concat([X, y], axis=1)).fit()
        # Return results
        models += [(f, model)]
    return models

# models with lowest RSS
def min_rss(statsmodels):
    return sorted(statsmodels, key=lambda tup: tup[1].ssr)[0]

# get all model results
model_subsets = []
for k in range(len(X.columns)):
    k=k+1
    subset = get_models(k, X, y)
    model_subsets += [subset]
    print('Progess: k = {}, done'.format(k))

best_subset_models = [min_rss(m) for m in model_subsets]
display(best_subset_models)


### Forward Stepwise ###
def forward_stepwise(X, y, results=[(0, [])]):
    # List predictors that havent's been used so far
    p_all = list(X.columns)
    p_used = results[-1][1]
    p_unused = [p for p in p_all if p not in p_used]
    
    # Job done, exit recursion
    if not p_unused:
        rss = [r[0] for r in results]
        preds = [r[1] for r in results]
        return pd.DataFrame({'rss': rss, 'predictors': preds}).drop(0).reset_index()
    
    # Get rss score for each possible additional predictor
    r = []
    for p in p_unused:
        f = 'CRIM ~ {}'.format('+'.join([p]+p_used))
        # Fit model
        model = smf.ols(formula=f, data=pd.concat([X, y], axis=1)).fit()
        r += [(model.ssr, [p]+p_used)]
    
    # Choose predictor which yields lowest rss
    min_rss = sorted(r, key=lambda tup: tup[0])[0]   
    new_results = results + [min_rss]
    # Recursive call to self
    return forward_stepwise(X, y, new_results)

forward_stepwise_results = forward_stepwise(X, y)
display(HTML('<h4>Forward Stepwise Selection</h4>'))
display(forward_stepwise_results)


### Backward Stepwise ###
def backward_stepwise(X, y, results=[]):
    
    p_all = list(X.columns)

    # Check if we're starting out here
    if not results:
        # Fit model with all features
        f = 'CRIM ~ {}'.format('+'.join(p_all))
        model = smf.ols(formula=f, data=pd.concat([X, y], axis=1)).fit()
        # Begin backward stepwise recursion
        return backward_stepwise(X, y, [(model.ssr, p_all)])
    else:
        p_used = results[-1][1]
    
    # Job done, exit recursion
    if len(p_used) == 1:
        rss   = [r[0] for r in results]
        preds = [r[1] for r in results]
        return pd.DataFrame({'rss': rss, 'predictors': preds})    
    
    # Get rss score for each possible removed predictor
    r = []
    for p in p_used:
        p_test = [i for i in p_used if i != p]
        f = 'CRIM ~ {}'.format('+'.join(p_test))
        # Fit model
        model = smf.ols(formula=f, data=pd.concat([X, y], axis=1)).fit()
        r += [(model.ssr, p_test)]
    
    # Choose removal of predictor which yields lowest rss
    min_rss = sorted(r, key=lambda tup: tup[0])[0]   
    new_results = results + [min_rss]
    return backward_stepwise(X, y, new_results)

backward_stepwise_results = backward_stepwise(X, y)
display(HTML('<h4>Backward Stepwise Selection</h4>'))
display(backward_stepwise_results)

print('Forward Stepwise Selection  : {}'.format(sorted(forward_stepwise_results.loc[5]['predictors'])))
print('Backward Stepwise Selection : {}'.format(sorted(backward_stepwise_results.loc[6]['predictors'])))



# b) compare the results of using the mathematical-adjustment approaches
# (Cp, AIC, BIC, & adjusted R2) to using 5-Fold Cross-Validation (5FCV)
# Propose a model (or set of models) that seem to perform well on
# this data set, and justify your answer.

# The 5FCV for forward and backward stepwise both yield the same model by
# choosing the model with the predictors B and LSTAT

### adj R2 ###
def max_adjr2(statsmodels):
    """Return model with lowest R-squared"""
    return sorted(statsmodels, reverse=True, key=lambda tup: tup[1].rsquared_adj)[0]

adjr2 = [max_adjr2(m)[1].rsquared_adj for m in model_subsets]
adjr2 = DataFrame(adjr2,columns=['adjusted R2'])
adjr2['parameters'] = adjr2.index + 1
adjr2['adjusted R2'].max()

sns.lineplot(x='parameters', y='adjusted R2', data = adjr2).set_title('Adj R-squared by number of predictors');
adjr2_models = [max_adjr2(m) for m in model_subsets]


### BIC ###
def min_bic(statsmodels):
    """Return model with lowest R-squared"""
    return sorted(statsmodels, reverse=False, key=lambda tup: tup[1].bic)[0]

bic = [min_bic(m)[1].bic for m in model_subsets]
bic = DataFrame(bic,columns=['bic'])
bic['parameters'] = bic.index + 1
bic['bic'].min()

sns.lineplot(x='parameters', y='bic', data = bic).set_title('BIC by number of predictors');


### AIC ###
def min_aic(statsmodels):
    """Return model with lowest R-squared"""
    return sorted(statsmodels, reverse=False, key=lambda tup: tup[1].aic)[0]

aic = [min_aic(m)[1].aic for m in model_subsets]
aic = DataFrame(aic,columns=['aic'])
aic['parameters'] = aic.index + 1
aic['aic'].min()

sns.lineplot(x='parameters', y='aic', data = aic).set_title('AIC by number of predictors');

### Cross Val ###
def mse(y_hat, y):
    """Calculate Mean Squared Error"""
    return np.sum(np.square(y_hat - y)) / y.size

def cross_val(formula, X, y, k):
    # Split dataset into k-folds
    # Note: np.array_split doesn't raise excpetion is folds are unequal in size
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    MSEs = []
    for f in np.arange(len(X_folds)):
        # Create training and test sets
        X_test  = X_folds[f]
        y_test  = y_folds[f]
        X_train = X.drop(X_folds[f].index)
        y_train = y.drop(y_folds[f].index)
        
        # Fit model
        model = smf.ols(formula=formula, data=pd.concat([X_train, y_train], axis=1)).fit()
        
        # Measure MSE
        y_hat = model.predict(X_test)
        MSEs += [mse(y_hat, y_test)]
    return (MSEs, formula)

### Forward Stepwise ###
forward_stepwise_subsets = forward_stepwise(X, y)
forward_stepwise_subsets['predictor_count'] = np.arange(1, 13)
display(forward_stepwise_subsets)

results = []
for preds in forward_stepwise_subsets['predictors']:
    f = 'CRIM ~ {}'.format(' + '.join(preds))
    results += [cross_val(f, X, y, 5)]

results_f_df = pd.DataFrame({'predictors': list(np.arange(1, len(results)+1)),
                           'MSE_mean': [np.mean(i[0]) for i in results],
                           'MSE_folds': [i[0] for i in results],
                           'Model': [i[1] for i in results]})
display(results_f_df)

sns.lineplot(x='predictors', y='MSE_mean', data=results_f_df).set_title('5FCV MSE by number of predictors');


### Backward Stepwise ###
backward_stepwise_subsets = backward_stepwise(X, y)
backward_stepwise_subsets['predictor_count'] = np.arange(1, 13)
display(backward_stepwise_subsets)

results = []
for preds in backward_stepwise_subsets['predictors']:
    f = 'CRIM ~ {}'.format(' + '.join(preds))
    results += [cross_val(f, X, y, 5)]

results_b_df = pd.DataFrame({'predictors': list(np.arange(1, len(results)+1)),
                           'MSE_mean': [np.mean(i[0]) for i in results],
                           'MSE_folds': [i[0] for i in results],
                           'Model': [i[1] for i in results]})
display(results_b_df)

sns.lineplot(x='predictors', y='MSE_mean', data=results_b_df);




