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

from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import 
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC

from scipy.stats import uniform 

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

x_test
x_test.dtypes

# What types are the variables 
bikes.dtypes
# Categorical = Season, holiday, workingday, weather...

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

clf = RF(n_estimators = 100, verbose = True )
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
train_sorted.head

bikes = pd.concat([train_sorted, test])

train_sorted.to_csv("train_sorted.csv")

#############################################################################
#      F E A T U R E       E N G I N E E R I N G
#############################################################################

# We combine datasets, so we only need to transform one and reduce risk of error. 

# Separate dates into date and hour
# What is the thought behind this? 
# While we are at it we should do day of year, day of month, month of year!

date = pd.DatetimeIndex(bikes['datetime'])
bikes['hour']      = date.hour
bikes["day.week"]  = date.dayofweek
bikes["day.month"] = date.day
bikes["month.year"]= date.month

bikes.dtypes

for col in ["season", "holiday", "workingday", "weather", "hour", "day.week", "day.month", "month.year"]:
    bikes[col] = bikes[col].astype('category')


bikes.describe()
# Put them back together for training and testing.

train_sorted =  # drop .10 least likely to be in test. 
train_sorted.drop("p", axis = 1, inplace= True)


Work = bikes[pd.notnull(bikes['count'])]
Train = Work.iloc[0:9799, ]
Train_Val = Work.iloc[9800:10886,]
Test = bikes[~pd.notnull(bikes['count'])]

y_count = Train["count"]
y_count_val = Train_Val["count"]

Test_Datetime = Test["datetime"]

features_to_drop  = ["count", "datetime", "p"]
Train.drop(features_to_drop,axis=1, inplace= True)
Train_Val.drop(features_to_drop,axis=1, inplace= True)
Test.drop(features_to_drop,axis=1, inplace=  True)

# What we are trying to predict! 

Train.dtypes
Train.head
numeric = ["atemp","humidity", "temp", "windspeed"]
normalized_df = (Train[numeric] - Train[numeric].min())/(Train[numeric].max() -Train[numeric].min())

#############################################################################
#     T R A I N 
#############################################################################

def rmsle1(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    
def rmsle2(y, y_, convertExp=False):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))


# --- Models --- 
model_svm = SVR(kernel='linear', C=1)
model_enet = ElasticNet(random_state=0)
model_rf = RandomForestRegressor()
model_knn = KNeighborsRegressor()
model_xgb = GradientBoostingRegressor()




# --- C V ---
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error

# ENET

modelenet = ElasticNetCV(cv = 10)
ENET_fit = modelenet.fit(Train, np.log1p(y_count))
preds = ENET_fit.predict(Train_Val)
rmsle1(np.exp(preds), y_count_val)


# RF
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model_rf = RandomizedSearchCV(estimator= model_rf, param_distributions=random_grid, n_iter=100, cv=3, 
                                verbose = 2, random_state = 42, n_jobs = -1)

rf_fit = model_rf.fit(Train, y_count)
preds = rf_fit.predict(Train_Val)
rmsle1(preds, y_count_val)
preds = rf_fit.predict(Test)
preds


# Prediction 

prediction1 = pd.DataFrame({"datetime": Test_Datetime, "count": np.exp(preds)})
cols = prediction1.columns.tolist()
cols = cols[-1:] + cols[:-1]
prediction1 = prediction1[cols]
prediction1.head
prediction1.to_csv("Prediction1.csv", index = False)

scipy.stats

prediction1 = pd.DataFrame({"datetime": Test_Datetime, "count": preds})
cols = prediction1.columns.tolist()
cols = cols[-1:] + cols[:-1]
prediction1 = prediction1[cols]
prediction1.head
prediction1.to_csv("Prediction2.csv", index = False)


