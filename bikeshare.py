import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn import preprocessing

from mlens.metrics import make_scorer
from mlens.metrics.metrics import rmse
from mlens.ensemble import SuperLearner

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn import cross_validation as CV
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC

#############################################################################
#      L O A D      D A T A     &    P R E P A R E
#############################################################################

train = pd.read_csv("bikeshareTRAIN.csv")
test = pd.read_csv("bikeshareTEST.csv")

print(train.describe())

train.head(3)

# Check for NA
train.isnull().sum()

# Assign indicator for AV. 

train['is_test'] = 0
test['is_test']  = 1

# combine dataframes for modeling later. 

data = pd.concat(( train, test ))

x = data.drop([ 'is_test', 'datetime', 'casual', 'registered', 'count'], axis = 1 )
y = data.is_test


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .9 )

# What types are the variables 
bikes.dtypes
# Categorical = Season, holiday, workingday, weather...

for col in ["season", "holiday", "workingday", "weather"]:
    bikes[col] = bikes[col].astype('category')


# Note, test has 3 more columns than test.
# This is because casual + registered = count 
# Count is the column we are trying to predict.
# I don't know why casual and registered are there.


#############################################################################
#      E X P L O R A T O R A Y       D A T A      A N A L Y S I S 
#############################################################################

sns.pairplot(train[["temp", "humidity", "windspeed", "atemp", "count"]])
sns.clustermap(train[["temp", "humidity", "windspeed", "atemp", "count"]].corr(), 
                center=0, linewidths=.5, figsize=(13, 13),vmax=.5, square=True,annot=True)


numeric_vars = train[["temp", "humidity", "windspeed", "atemp", "count"]]
log_num = numeric_vars[["windspeed", "count"]].apply(np.log1p)
log_num.columns = ['log_wind', 'log_count']
log_num
numeric_vars = pd.concat([log_num, numeric_vars], axis = 1)
numeric_vars
numeric_vars = preprocessing.scale(numeric_vars)
numeric_vars = pd.DataFrame(numeric_vars)

numeric_vars.head()
numeric_vars.columns = ["log_wind","log_count", "temp", "humidity", "windspeed", "atemp" , "count"]

sns.boxplot(data = numeric_vars )




#----------------------------
#   ADVERSARIAL VALIDATION
#----------------------------

# --- RF ---
print "training random forest..."

n_estimators = 100
clf = RF(n_estimators = n_estimators, n_jobs = -1, verbose = True )
clf.fit(x_train, y_train)

# predict

p = clf.predict_proba( x_test )[:,1]
auc = AUC(y_test, p )
print "AUC: {:.2%}".format(auc)

# There is definite separability between training and test. 

#----------------------------
#   SORT
#----------------------------
train_sorted = train.drop([ 'is_test', 'datetime', 'casual', 'registered', 'count'], axis = 1 )

# So we apply the model to predict what data is likeliest to resemble the test data
# To the training dataframe. 
train_sorted['p'] = clf.predict_proba(train_sorted)[:,1]

# Need to add variables back 

train_sorted = pd.concat([train_sorted, train[['datetime','count']]], axis = 1)

train_sorted.sort_values('p', ascending= False, inplace= True)

train_sorted = train_sorted.iloc[0:9799, ] # drop .10 least likely to be in test. 
train_sorted.drop("p", axis = 1, inplace= True)

bikes = pd.concat([train_sorted, test])

#############################################################################
#      F E A T U R E       E N G I N E E R I N G
#############################################################################

# We combine datasets, so we only need to transform one and reduce risk of error. 

# Separate dates into date and hour
# What is the thought behind this? 
# While we are at it we should do day of year, day of month, month of year!

date = pd.DatetimeIndex(bikes['datetime'])
bikes['hour']      = date.time
bikes["day.week"]  = date.dayofweek
bikes["day.month"] = date.day
bikes["month.year"]= date.month






dropping = ['casual', 'registered']
bikes.drop(dropping,  axis = 1, inplace = True)

bikes.head()

# Put them back together for training and testing.

Train = bikes[pd.notnull(bikes['count'])].sort_values(by=["datetime"])
Test = bikes[~pd.notnull(bikes['count'])].sort_values(by=["datetime"])

y_count = Train["count"]

features_to_drop  = ["count","datetime","date"]
Train.drop(features_to_drop,axis=1, inplace= True)
Test.drop(features_to_drop,axis=1, inplace=  True)

 # What we are trying to predict! 

Train.dtypes
Train.head

#############################################################################
#     T R A I N 
#############################################################################

def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


# --- Models --- 
model.svm = SVR(kernel='linear', C=1)
model.enet
model.knn



# --- C V ---
from sklearn.model_selection import cross_val_score


scores.svm = cross_val_score(clf, iris.data, iris.target, cv=5)

scores = cross_validate(lasso, X, y,
...                         scoring=('r2', 'neg_mean_squared_error'))


# --- Build ---
# Passing a scoring function will create cv scores during fitting
# the scorer should be a simple function accepting to vectors and returning a scalar
ensemble = SuperLearner(scorer= rmsle, random_state = 42, verbose=2)

# Build the first layer
ensemble.add([RandomForestRegressor(random_state=42),
                    KNeighborsRegressor(), LinearRegression()])

# Attach the final meta estimator
ensemble.add_meta(LinearRegression())

# --- Use ---

# Fit ensemble
ensemble.fit(Train)

# Predict
preds = ensemble.predict(X[75:])

print("Fit data:\n%r" % ensemble.data)

