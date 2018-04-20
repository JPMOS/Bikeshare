import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import scikitlearn 

train = pd.read_csv("bikeshareTRAIN.csv")
test = pd.read_csv("bikeshareTEST.csv")

print(train.shape)
print(test.shape)

print(train.describe())

train.head(3)
train.isnull().sum()

# Separate dates into date and hour
date = pd.DatetimeIndex(train['datetime'])
train['date'] = date.date
train['hour'] = date.time
del train['datetime']
train.head()

# Separate dates into date and hour
date = pd.DatetimeIndex(test['datetime'])
test['date'] = date.date
test['hour'] = date.time
del test['datetime']
test.head()

# 1. combine dataframes for modeling later. 
bikes = pd.concat([train, test])

# 2. separate numerical from categorical.

bikes.astype()
print(bikes[0:4,0:4])


sns.pairplot(train)
plt.show()

# correlation heatmap
sns.clustermap(train.corr(), center=0, linewidths=.5, figsize=(13, 13), vmax=.5, square=True,annot=True)
