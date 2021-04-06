# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from r_wrapper import diftrans
from r_wrapper import base
from r_wrapper import stats
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

path = os.path.join(os.getcwd())
Beijing_sample = pd.read_csv(os.path.join(path,'Beijing_sample.csv'))
Tianjin_sample = pd.read_csv(os.path.join(path,'Tianjin_sample.csv'))

Beijing_sample = base.get("Beijing_sample")
Tianjin_sample = base.get("Tianjin_sample")


### Exercise 3.1. For each of the following, ensure that the first column is 
### MSRP and the second column is count

# keep 2010 and 2011 data only
Beijing = Beijing_sample[(Beijing_sample['year']>= 2010) & 
                         (Beijing_sample['year'] < 2012)]

# collect unique MSRP values
uniqueMSRP = pd.DataFrame(Beijing.MSRP.unique()).rename(columns={0:'MSRP'})

# aggregate sales at each price for 2010 (pre-lottery)
Beijing10_sales = Beijing[(Beijing['year'] == 2010)].groupby('MSRP').aggregate({'sales':[sum]})
Beijing10_sales = Beijing10_sales.unstack().reset_index().rename_axis(None, axis=1)
Beijing10_sales = Beijing10_sales.drop(columns=['level_0',
                                                'level_1']).rename(columns={0:'count'})
# merge the MSRP and sales
Beijing_pre = uniqueMSRP.merge(Beijing10_sales, how='left', on ="MSRP")
Beijing_pre[['count']] = Beijing_pre[['count']].fillna(value=0)
Beijing_pre = Beijing_pre.sort_values('MSRP') 
df2 = Beijing_pre.pop('count') # uncount

Beijing_distribution_pre = pd.DataFrame(Beijing_pre.values.repeat(df2, axis=0), 
                                        columns=Beijing_pre.columns)


## a. Clean data of Beijing car sales in 2011, and store the data frame in a 
## variable called Beijing_post.
# aggregate sales at each price for 2011 (post-lottery)
Beijing11_sales = Beijing[(Beijing['year'] == 2011)].groupby('MSRP').aggregate({'sales':[sum]})
Beijing11_sales = Beijing11_sales.unstack().reset_index().rename_axis(None, axis=1)
Beijing11_sales = Beijing11_sales.drop(columns=['level_0',
                                                'level_1']).rename(columns={0:'count'})
# merge the MSRP and sales
Beijing_post = uniqueMSRP.merge(Beijing11_sales, how='left', on ="MSRP")
Beijing_post[['count']] = Beijing_post[['count']].fillna(value=0)
Beijing_post = Beijing_post.sort_values('MSRP') 
Beijing_post.head() #preview data
df3 = Beijing_post.pop('count') # uncount

Beijing_distribution_post = pd.DataFrame(Beijing_post.values.repeat(df3, axis=0),
                                         columns=Beijing_post.columns)


## b. Clean data of Tianjin car sales in 2010 as a variable called Tianjin_pre.
# keep 2010 and 2011 data only
Tianjin = Tianjin_sample[(Tianjin_sample['year']>= 2010) & 
                         (Tianjin_sample['year'] < 2012)]

# collect unique MSRP values
uniqueMSRP = pd.DataFrame(Tianjin.MSRP.unique()).rename(columns={0:'MSRP'})

# aggregate sales at each price for 2010 (pre-lottery)
Tianjin10_sales = Tianjin[(Tianjin['year'] == 2010)].groupby('MSRP').aggregate({'sales':[sum]})
Tianjin10_sales = Tianjin10_sales.unstack().reset_index().rename_axis(None, axis=1)
Tianjin10_sales = Tianjin10_sales.drop(columns=['level_0',
                                                'level_1']).rename(columns={0:'count'})
# merge the MSRP and sales
Tianjin_pre = uniqueMSRP.merge(Tianjin10_sales, how='left', on ="MSRP")
Tianjin_pre[['count']] = Tianjin_pre[['count']].fillna(value=0)
Tianjin_pre = Tianjin_pre.sort_values('MSRP') 
Tianjin_pre.head() #preview data
df4 = Tianjin_pre.pop('count') # uncount

Tianjin_distribution_pre = pd.DataFrame(Tianjin_pre.values.repeat(df4, axis=0), 
                                        columns=Tianjin_pre.columns)


## c. Clean data of Tianjin car sales in 2011 as a variable called Tianjin_post.
# aggregate sales at each price for 2011 (post-lottery)
Tianjin11_sales = Tianjin[(Tianjin['year'] == 2011)].groupby('MSRP').aggregate({'sales':[sum]})
Tianjin11_sales = Tianjin11_sales.unstack().reset_index().rename_axis(None, axis=1)
Tianjin11_sales = Tianjin11_sales.drop(columns=['level_0',
                                                'level_1']).rename(columns={0:'count'})
# merge the MSRP and sales
Tianjin_post = uniqueMSRP.merge(Tianjin11_sales, how='left', on ="MSRP")
Tianjin_post[['count']] = Tianjin_post[['count']].fillna(value=0)
Tianjin_post = Tianjin_post.sort_values('MSRP') 
Tianjin_post.head() #preview data
df5 = Tianjin_post.pop('count') # uncount

Tianjin_distribution_post = pd.DataFrame(Tianjin_post.values.repeat(df5, axis=0), 
                                        columns=Tianjin_post.columns)



### Exercise 3.2. Replicate Figure 1 for Tianjin.
## a. Overlay the histograms that describe the 2010 and 2011 distribution of 
## Tianjin car sales. Be sure to normalize the histograms so the area of the 
## bars in each histogram sum to 1.
fig, ax = plt.subplots()
for a in [Beijing_distribution_pre, Beijing_distribution_post]:
    sns.distplot(a/1000, ax=ax, kde=True)
plt.xlabel("MSRP(1000RMB)", size=11)
plt.ylabel("Density", size=11)
plt.title("Pre-lottery (blue) vs. Post-lottery (brown)\n Sales Distributions of Beijing Cars", 
          size=13)

fig, ax = plt.subplots()
for a in [Tianjin_distribution_pre, Tianjin_distribution_post]:
    sns.distplot(a/1000, ax=ax, kde=True)
plt.xlabel("MSRP(1000RMB)", size=11)
plt.ylabel("Density", size=11)
plt.title("Pre-lottery (blue) vs. Post-lottery (brown)\n Sales Distributions of Tianjin Cars", 
          size=13)

## b. Compare and contrast the shift between the Beijing distributions with 
## the shift between the Tianjin distributions. Based on the shift in Tianjin 
## carsales, should we be surprised to see the shift in Beijing car sales? 




### Exercise 3.3. 
## a. Run the preceding code block so you have access to placebo_1.
base.set_seed(1)

uniqueMSRP = pd.DataFrame(Beijing.MSRP.unique()).rename(columns={0:'MSRP'})

Beijing_pre = uniqueMSRP.merge(Beijing10_sales, how='left', on ="MSRP")
Beijing_pre[['count']] = Beijing_pre[['count']].fillna(value=0)
Beijing_pre = Beijing_pre.sort_values('MSRP') 

count = stats.rmultinom(n = 1, size = sum(Beijing_pre['count']),
                        prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_1 = pd.DataFrame(data=d)
print(placebo_1)
print(placebo_1.dtypes)


## b. Use rmultinom to sample observations from Beijing_pre. Store the 
## resulting data frame in placebo_2. Be careful to draw the correct 
## number of observations.
base.set_seed(1)

Beijing_post = uniqueMSRP.merge(Beijing11_sales, how='left', on ="MSRP")
Beijing_post[['count']] = Beijing_post[['count']].fillna(value=0)
Beijing_post = Beijing_post.sort_values('MSRP') 

count = stats.rmultinom(n = 1, size = sum(Beijing_post['count']),
                        prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_2 = pd.DataFrame(data=d)
print(placebo_2)


## c. Compare placebo_1 and placebo_2. Do they appear to be drawn from the 
## same distribution?
df_placebo1 = placebo_1.pop('count')
placebo1_distribution = pd.DataFrame(placebo_1.values.repeat(df_placebo1, axis=0), 
                                     columns=placebo_1.columns)

df_placebo2 = placebo_2.pop('count')
placebo2_distribution = pd.DataFrame(placebo_2.values.repeat(df_placebo2, axis=0), 
                                     columns=placebo_2.columns)

fig, ax = plt.subplots()
for a in [placebo1_distribution, placebo2_distribution]:
    sns.distplot(a/1000, ax=ax, kde=True)
plt.xlabel("MSRP(1000RMB)", size=11)
plt.ylabel("Density", size=11)
plt.title("Placebo_1 (blue) vs. Placebo_2 (brown)\n Sales Distributions of Beijing Cars", size=13)



### Exercise 3.4. 
## a. Compute the transport cost between the two placebo distributions for 
## different values of 𝑑 from 0 to 100,000
base.set_seed(1)
count = stats.rmultinom(n = 1, size = sum(Beijing_pre['count']),
                        prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_1 = pd.DataFrame(data=d)

base.set_seed(1)
count = stats.rmultinom(n = 1, size = sum(Beijing_post['count']),
                        prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_2 = pd.DataFrame(data=d)

placebo_at_100k = diftrans.diftrans(pre_main = placebo_1, 
                                    post_main = placebo_2, 
                                    bandwidth_seq = np.arange(0,100000,step=1000))
print(placebo_at_100k)


## b. For the same values of 𝑑, compute the transport cost between the 
## observed distributions for 2010 and 2011 Beijing car sales. 
observed_at_100k = diftrans.diftrans(pre_main = Beijing_pre, 
                                    post_main = Beijing_post, 
                                    bandwidth_seq = np.arange(0,100000,step=1000))
print(observed_at_100k)


## c. Plot the placebo costs and the empirical costs obtained in the previous 
## two steps with the bandwidth as the x-axis.
plt.plot('bandwidth', 'main', label = "Placebo", data = placebo_at_100k) 
plt.plot('bandwidth', 'main', label = "Observed", data = observed_at_100k)
plt.xlabel("bandwidth", size=11)
plt.ylabel("cost", size=11)
plt.title("Placebo vs. Empirical Costs of Beijing Car Sales 2010~2011", size=13)
plt.legend() 
plt.show();

## d. For which values of 𝑑 is the placebo cost less than 0.05%?
lessthan05 = placebo_at_100k[placebo_at_100k['main'] < 0.0005] 
lessthan05.head()


## e. For the smallest value of 𝑑 found in the previous step, what is the 
## empirical transport cost? This estimate for the lower bound on the volume of 
## black market transactions is what we call the before-and-after estimate.
print(observed_at_100k[observed_at_100k['bandwidth'] == 25000])



### Exercise 3.5. 
## a. Compute the (3) for different values of 𝑑 from 0 to 50,000. 
## Unlike before, we go up to 50,000 because we are using the conservative 
# bandwidth of 2𝑑𝑑 for the Beijing transport cost.
uniqueMSRP = pd.DataFrame(Tianjin.MSRP.unique()).rename(columns={0:'MSRP'})

Tianjin_pre = uniqueMSRP.merge(Tianjin10_sales, how='left', on ="MSRP")
Tianjin_pre[['count']] = Tianjin_pre[['count']].fillna(value=0)
Tianjin_pre = Tianjin_pre.sort_values('MSRP') 

Tianjin_post = uniqueMSRP.merge(Tianjin11_sales, how='left', on ="MSRP")
Tianjin_post[['count']] = Tianjin_post[['count']].fillna(value=0)
Tianjin_post = Tianjin_post.sort_values('MSRP') 

dit_at_50k = diftrans.diftrans(pre_main = Beijing_pre,
                             post_main = Beijing_post,
                             pre_control = Tianjin_pre,
                             post_control = Tianjin_post,
                             bandwidth_seq = np.arange(0,50000,step=1000),
                             conservative = True)
print(dit_at_50k)




## b. Using what you learned from Exercise 3.3, construct a placebo distribution 
## that is sampled from Beijing_pre whose size is the number of Beijing cars in 
## 2010. Call this distribution placebo_Beijing_1.
base.set_seed(1)
count = stats.rmultinom(n = 1, size = sum(Beijing_pre['count']),
                        prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_Beijing_1 = pd.DataFrame(data=d)


## c. Construct another placebo distribution called placebo_Beijing_2 that is 
## also sampled from Beijing_pre but is of size is the number of Beijing cars in 2011.
base.set_seed(1)
count = stats.rmultinom(n = 1, size = sum(Beijing_post['count']),
                        prob = Beijing_pre['count'])
count2 = count[:,0]
d = {'MSRP': Beijing_pre['MSRP'], 'count' : count2}
placebo_Beijing_2 = pd.DataFrame(data=d)


## d. Construct a placebo distribution called placebo_Tianjin_1 that is sampled 
## from Tianjin_pre and whose size is the number of Tianjin cars in 2010.
base.set_seed(1)
count = stats.rmultinom(n = 1, size = sum(Tianjin_pre['count']),
                        prob = Tianjin_pre['count'])
count2 = count[:,0]
d = {'MSRP': Tianjin_pre['MSRP'], 'count' : count2}
placebo_Tianjin_1 = pd.DataFrame(data=d)


## e. Construct a placebo distribution called placebo_Tianjin_2 that is sampled 
## from Tianjin_pre and whose size is the number of Tianjin cars in 2011
base.set_seed(1)
count = stats.rmultinom(n = 1, size = sum(Tianjin_post['count']),
                        prob = Tianjin_pre['count'])
count2 = count[:,0]
d = {'MSRP': Tianjin_pre['MSRP'], 'count' : count2}
placebo_Tianjin_2 = pd.DataFrame(data=d)


## f. Using the four placebo distributions, compute the placebo counterpart of 
## (3) for the same values of 𝑑 that you used in part a.
placebo_dit_at_50k = diftrans.diftrans(pre_main = placebo_Beijing_1,
                             post_main = placebo_Beijing_2,
                             pre_control = placebo_Tianjin_1,
                             post_control = placebo_Tianjin_2,
                             bandwidth_seq = np.arange(0,50000,step=1000),
                             conservative = True)
print(placebo_dit_at_50k)


## g. Create a plot of the absolute value of the placebo differences-in-transports 
## estimator on the y-axis and the bandwidth on the x-axis.
plt.plot(placebo_dit_at_50k['bandwidth'], np.absolute(placebo_dit_at_50k['diff2d'])) 
plt.xlabel("bandwidth", size=11)
plt.ylabel("diff2d", size=11)
plt.title("Placebo differences in transports Beijing and Tianjin", size=13)
plt.show();


## h. For which values of 𝑑 does the absolute value of the placebo differences
## -in-transports estimator stay below 0.05%? Note that the absolute difference
## is not a monotonically decreasing object, so this difference may even increase 
## as we increase the bandwidth. Temporary increases above the 0.05% threshold 
## can be ignored.
lessthan05_2 = placebo_dit_at_50k[np.absolute(placebo_dit_at_50k['diff2d']) < 0.0005] 
lessthan05_2.head()


## i. Among all the values of 𝑑 that you found in the previous step, which one 
## yielded the largest value of (3) from part a? This is the difference-in-transports 
## estimator.
print(dit_at_50k[dit_at_50k.diff2d == dit_at_50k.diff2d.max()]) 
