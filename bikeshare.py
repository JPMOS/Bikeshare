import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import scikitlearn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
%matplotlib inline 

train = pd.read_csv("bikeshareTRAIN.csv")
test = pd.read_csv("bikeshareTEST.csv")

print(train.shape)
print(test.shape)

print(train.describe())

train.head(3)
train.isnull().sum()

# Separate dates into separate columns
datetime = test['datetime']
date = pd.DatetimeIndex(train['datetime'])
train["dayofweek"] = date.dayofweek
train["hour"] = date.hour
train["month"] = date.month
train['year']= date.year

date = pd.DatetimeIndex(test['datetime'])
test["dayofweek"] = date.dayofweek
test["hour"] = date.hour
test["month"] = date.month
test['year']= date.year
test= test.drop(['datetime'],axis=1)

train.head()
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

X=train.drop(['casual','registered','count','datetime'],axis=1)
y=train['count']

# Split train set into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 10)

# Linear regression (needs work, MSE too big)
lm1 = LinearRegression()
lm1.fit(X_train, y_train)
lmpredict = lm1.predict(X_test)
mean_squared_error(y_test, lmpredict)

# Random forest (also needs work)
rf1 = RandomForestRegressor()
rf1.fit(X_train, y_train)
rfpredict = rf1.predict(X_test)
mean_squared_error(y_test, rfpredict)

rftestpredict = rf1.predict(test)

bike_predict=pd.DataFrame({'lmcount':lmtestpredict, 'rfcount':rftestpredict, 'datetime':datetime})
bike_predict
