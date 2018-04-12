import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import scikitlearn 

train = pd.read_csv("bikeshareTRAIN.csv")
test = pd.read_csv("bikeshareTEST.csv")

print(train.describe())

train.head(3)
train.isnull().sum()

# 1. combine dataframes for modeling later. 
bikes = pd.concat([train, test])

# 2. separate numerical from categorical.

bike.astype()
print(bikes[0:4,0:4])


sns.pairplot(train)
plt.show()

