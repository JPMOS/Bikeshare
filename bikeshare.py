import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV

from mlens.metrics import make_scorer
from mlens.metrics.metrics import rmse
from mlens.ensemble import SuperLearner

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#############################################################################
#      L O A D      D A T A     &    P R E P A R E
#############################################################################

train = pd.read_csv("bikeshareTRAIN.csv")
test = pd.read_csv("bikeshareTEST.csv")

print(train.describe())

train.head(3)
train.isnull().sum()

# 1. combine dataframes for modeling later. 
bikes = pd.concat([train, test])

# What types are the variables 
bikes.dtypes
# Categorical = Season, holiday, workingday, weather...

for col in ["season", "holiday", "workingday", "weather"]:
    bikes[col] = bikes[col].astype('category')

# Note, test has 3 more columns than test.
# This is because casual + registered = count 
# Count is the column wer are trying to predict.
# I don't know why casual and registered are there.


#############################################################################
#      E X P L O R A T O R A Y       D A T A      A N A L Y S I S 
#############################################################################

sns.pairplot(train[["temp", "humidity", "windspeed", "atemp", "count"]])
sns.clustermap(train[["temp", "humidity", "windspeed", "atemp", "count"]].corr(), 
                center=0, linewidths=.5, figsize=(13, 13),vmax=.5, square=True,annot=True)



#############################################################################
#      F E A T U R E       E N G I N E E R I N G
#############################################################################

# We combine datasets, so we only perform operation on one at a time. 

# Separate dates into date and hour
# What is the thought behind this? 
# While we are at it we should do day of year, day of month, month of year!

date = pd.DatetimeIndex(bikes['datetime'])
bikes['date']      = date.date
bikes['hour']      = date.time
bikes["day.week"]  = date.dayofweek
bikes["day.month"] = date.day
bikes["month.year"]= date.month


bikes.head()


# Put them back together for training and testing.

Train = bikes[pd.notnull(bikes['count'])].sort_values(by=["datetime"])
Test = bikes[~pd.notnull(bikes['count'])].sort_values(by=["datetime"])

del Train['datetime']
del Test['datetime']

Train.dtypes

#############################################################################
#     T R A I N 
#############################################################################

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


# --- Build ---
# Passing a scoring function will create cv scores during fitting
# the scorer should be a simple function accepting to vectors and returning a scalar
ensemble = SuperLearner(scorer= rmsle, random_state = 42, verbose=2)

# Build the first layer
ensemble.add([RandomForestRegressor(random_state=42), SVR(),
                    KNeighborsRegressor(), LinearRegression()])

# Attach the final meta estimator
ensemble.add_meta(LinearRegression())

# --- Use ---

# Fit ensemble
ensemble.fit(Train)

# Predict
preds = ensemble.predict(X[75:])

print("Fit data:\n%r" % ensemble.data)
