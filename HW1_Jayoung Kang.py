# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 22:43:10 2021

@author: 13124
"""
import os
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt


### c) Question 10
## (a) To begin, load in the Boston data set. 
path = os.path.join(os.getcwd())
df = pd.read_csv(os.path.join(path,'Boston.csv'))
print(df.shape)


## 10 (b) 
sns.pairplot(df)


## 10 (c) 
sns.pairplot(df, vars = ["CRIM", "AGE", "DIS", "MDEV"]);

for col in df.iloc[:,1:15].columns:
    sns.scatterplot(df[col],df['CRIM'])
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('CRIM')
    plt.show()


## 10 (d) 
def plot_hist(df, var):
    fig, ax = plt.subplots()
    df[var].hist(); 
    ax.set_title(var)
    plt.show()

plot_hist(df, 'CRIM')
# crim: most suburbs (400+) have low crime rate around 0 and 10 but there is a 
# long tail of suburbs with particularly high crime rate of between 20 and 80
plot_hist(df, 'TAX')
# tax: there is a clear divide between those with tax rate between a group 
# around 200 and 500 and another group with particularly high tax rate of
# around 650 and 700  
plot_hist(df, 'PTRATIO')
# ptratio: the distribution is generally skewed to the left and there is a high
# number of suburbs (175+) that have ptratio of above 20 but they do not seem
# particularly high relative to the general distribution of between 12.6 to 22


## 10 (e) 
df[df['CHAS']==1].count()
# 35 suburbs


## 10 (f) 
df['PTRATIO'].median(axis=0)
# 19.05


## 10 (g) 
mdev_min = df[df.MDEV==df.MDEV.min()]
mdev_min = mdev_min.T
print(mdev_min)
# Suburbs 398 and 405 have the lowest MDEV
desc = df.describe()
# crim: both above 75th percentile
# zn: both at min
# indus: both at 75th percentile
# chas: both at min
# nox: both above 75th percentile
# rm: both below 25th percentile
# age: both at max
# dis: both below 25th percentile
# rad: both at max
# tax: both at 75th percentile
# ptratio: both at 75th percentile
# b: at max for 398 and between 25th and 50th percentile for 405 
# lstat: both above 75th percentile


## 10 (h) 
df[df['RM']>7].count()
# More than 7 rooms: 64
df[df['RM']>8].count()
# More than 8 rooms: 13
desc_rm8=df[df['RM']>8].describe()
# These suburbs have lower average of crime rate, higher average of median 
# value of owner occupied homes and lower average of percentage of lower status
# population than the total data average 



#### 2. Questions from Chapter 3 of the Introduction to Statistical Learning

### a) Question 3
## (a) Which answer is correct, and why?
# (iii) The FOC when deriving the model by gender shows 35-10*GPA meaning that
# if GPA is high enough, women earn less than men on average. 


## (b) Predict the salary of a female with IQ of 110 and a GPA of 4.0
print(50 + 4*20 + 110*0.07 + 1*35 + 0.01*4*110 - 10*4*1)
# She is expected to earn 137.1k dollars


## (c) True or false: Since the coefficient for the GPA/IQ interaction
## term is very small, there is very little evidence of an interaction
## effect. Justify your answer.
# False. Evidence for interaction effect is measured by the t-statistics or 
# the p-value. 


### Question 15(a) 
res_zn = smf.ols('CRIM ~ ZN', data = df).fit()
print(res_zn.summary())
# yes, negative

res_indus = smf.ols('CRIM ~ INDUS', data = df).fit()
print(res_indus.summary())
# yes, positive

res_chas = smf.ols('CRIM ~ CHAS', data = df).fit()
print(res_chas.summary())
# no, p-value 0.214

res_nox = smf.ols('CRIM ~ NOX', data = df).fit()
print(res_nox.summary())
# yes, positive

res_rm = smf.ols('CRIM ~ RM', data = df).fit()
print(res_rm.summary())
# yes, negative

res_age = smf.ols('CRIM ~ AGE', data = df).fit()
print(res_age.summary())
# yes, positive

res_dis = smf.ols('CRIM ~ DIS', data = df).fit()
print(res_dis.summary())
# yes, negative

res_rad = smf.ols('CRIM ~ RAD', data = df).fit()
print(res_rad.summary())
# yes, positive

res_tax = smf.ols('CRIM ~ TAX', data = df).fit()
print(res_tax.summary())
# yes, positive

res_ptratio = smf.ols('CRIM ~ PTRATIO', data = df).fit()
print(res_ptratio.summary())
# yes, positive

res_b = smf.ols('CRIM ~ B', data = df).fit()
print(res_b.summary())
# yes, negative

res_lstat = smf.ols('CRIM ~ LSTAT', data = df).fit()
print(res_lstat.summary())
# yes, positive

res_mdev = smf.ols('CRIM ~ MDEV', data = df).fit()
print(res_mdev.summary())
# yes, negative

# plot regression
def plot_results(crim, var):
    plt.figure(figsize=(12,8))
    sns.regplot(x=var, y=crim, data=df, 
                scatter_kws={"color": "blue"}, line_kws={"color": "red"})
    plt.title(var, fontsize=20)
    plt.xlabel(var, fontsize=15)
    plt.ylabel(crim, fontsize=15)
    plt.show()

plot_results('CRIM', 'CHAS')
plot_results('CRIM', 'AGE')
plot_results('CRIM', 'DIS')
plot_results('CRIM', 'MDEV')


## 15 (b) 
res_all = smf.ols('CRIM ~ ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT + MDEV', 
                  data = df).fit()
print(res_all.summary())


## 15(c)
# There are fewer significant variables in (b). 
# reference: https://stackoverflow.com/questions/47388258/how-to-extract-the-regression-coefficient-from-statsmodels-api

res_zn_t = pd.read_html(res_zn.summary().tables[1].as_html(),
                        header=0,index_col=0)[0]

res_indus_t = pd.read_html(res_indus.summary().tables[1].as_html(),
                           header=0,index_col=0)[0]

res_chas_t = pd.read_html(res_chas.summary().tables[1].as_html(),
                          header=0,index_col=0)[0]

res_nox_t = pd.read_html(res_nox.summary().tables[1].as_html(),
                         header=0,index_col=0)[0]

res_rm_t = pd.read_html(res_rm.summary().tables[1].as_html(),
                        header=0,index_col=0)[0]

res_age_t = pd.read_html(res_age.summary().tables[1].as_html(),
                         header=0,index_col=0)[0]

res_dis_t = pd.read_html(res_dis.summary().tables[1].as_html(),
                         header=0,index_col=0)[0]

res_rad_t = pd.read_html(res_rad.summary().tables[1].as_html(),
                         header=0,index_col=0)[0]

res_tax_t = pd.read_html(res_tax.summary().tables[1].as_html(),
                         header=0,index_col=0)[0]

res_ptratio_t = pd.read_html(res_ptratio.summary().tables[1].as_html(),
                             header=0,index_col=0)[0]

res_b_t = pd.read_html(res_b.summary().tables[1].as_html(),
                       header=0,index_col=0)[0]

res_lstat_t = pd.read_html(res_lstat.summary().tables[1].as_html(),
                           header=0,index_col=0)[0]

res_mdev_t = pd.read_html(res_mdev.summary().tables[1].as_html(),
                          header=0,index_col=0)[0]

res_all_t = pd.read_html(res_all.summary().tables[1].as_html(),
                          header=0,index_col=0)[0]


d = {'uni': [res_zn_t['coef'].values[1],
             res_indus_t['coef'].values[1],
             res_chas_t['coef'].values[1],
             res_nox_t['coef'].values[1],
             res_rm_t['coef'].values[1],
             res_age_t['coef'].values[1],
             res_dis_t['coef'].values[1],
             res_rad_t['coef'].values[1],
             res_tax_t['coef'].values[1],
             res_ptratio_t['coef'].values[1],
             res_b_t['coef'].values[1],
             res_lstat_t['coef'].values[1],
             res_mdev_t['coef'].values[1]],
     'multi': res_all_t['coef'].values[1:14]}

data = pd.DataFrame(data=d,
                    index = df.columns.tolist()[1:15])

data.columns.tolist()[1:15]

plt.scatter(data['uni'], data['multi'])
plt.title("Multivariate vs. Univariate")
plt.xlabel("Univariate Reg Coefficient")
plt.ylabel("Multivariate Reg Coefficient")
plt.show()


## 15(d) 
res_zn3 = smf.ols('CRIM ~ ZN + I(ZN**2) + I(ZN**3)', data = df).fit()
print(res_zn3.summary())

res_indus3 = smf.ols('CRIM ~ INDUS + I(INDUS**2) + I(INDUS**3)', 
                     data = df).fit()
print(res_indus3.summary())

res_chas3 = smf.ols('CRIM ~ CHAS + I(CHAS**2) + I(CHAS**3)', data = df).fit()
print(res_chas3.summary())

res_nox3 = smf.ols('CRIM ~ NOX + I(NOX**2) + I(NOX**3)', data = df).fit()
print(res_nox3.summary())

res_rm3 = smf.ols('CRIM ~ RM + I(RM**2) + I(RM**3)', data = df).fit()
print(res_rm3.summary())

res_age3 = smf.ols('CRIM ~ AGE + I(AGE**2) + I(AGE**3)', data = df).fit()
print(res_age3.summary())

res_dis3 = smf.ols('CRIM ~ DIS + I(DIS**2) + I(DIS**3)', data = df).fit()
print(res_dis3.summary())

res_rad3 = smf.ols('CRIM ~ RAD + I(RAD**2) + I(RAD**3)', data = df).fit()
print(res_rad3.summary())

res_tax3 = smf.ols('CRIM ~ TAX + I(TAX**2) + I(TAX**3)', data = df).fit()
print(res_tax3.summary())

res_ptratio3 = smf.ols('CRIM ~ PTRATIO + I(PTRATIO**2) + I(PTRATIO**3)', 
                       data = df).fit()
print(res_ptratio3.summary())

res_b3 = smf.ols('CRIM ~ B + I(B**2) + I(B**3)', data = df).fit()
print(res_b3.summary())

res_lstat3 = smf.ols('CRIM ~ LSTAT + I(LSTAT**2) + I(LSTAT**3)', 
                     data = df).fit()
print(res_lstat3.summary())

res_mdev3 = smf.ols('CRIM ~ MDEV + I(MDEV**2) + I(MDEV**3)', data = df).fit()
print(res_mdev3.summary())


