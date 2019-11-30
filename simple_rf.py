#MSBD 5001 individual project
#An Weigang
#model1 - RandomForest

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import datetime
import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('train.csv', low_memory=False)
test_data =  pd.read_csv('test.csv', low_memory=False)

#print(train_data.isnull().sum())

#Data progressing of training data
train_data = train_data.drop(columns=['is_free'])
train_data['purchase_date'] = pd.to_datetime(train_data['purchase_date']).apply(lambda x: x.date())
train_data['release_date'] = pd.to_datetime(train_data['release_date']).apply(lambda x: x.date())
train_data['diff'] = train_data['purchase_date'] - train_data['release_date']
train_data['pmonth'] = train_data['purchase_date'].apply(lambda dt: dt.month)
train_data['pday'] = train_data['purchase_date'].apply(lambda dt: dt.day)
train_data['rmonth'] = train_data['release_date'].apply(lambda dt: dt.month)
train_data['rday'] = train_data['release_date'].apply(lambda dt: dt.day)
train_data.fillna(0, inplace=True)
#Count difference between purchase_date and release_date by day. 
train_data['diff'] = train_data['diff'].apply(lambda x: x.total_seconds()/86400)
train_data['rate'] = train_data['total_positive_reviews'] / train_data['total_negative_reviews']
train_data['rate'] = train_data['rate'].apply(lambda x: 0 if(x == np.inf) else x)
train_data['previews_rate'] = train_data['total_positive_reviews'] / (train_data['total_positive_reviews'] + train_data['total_negative_reviews'])
train_data['nreviews_rate'] = train_data['total_negative_reviews'] / (train_data['total_positive_reviews'] + train_data['total_negative_reviews'])
train_data = train_data.drop(columns=['purchase_date', 'release_date'])
train_data = train_data.drop(columns=['genres','categories', 'tags'])
mean1 = train_data.mean()
train_data.fillna(mean1, inplace=True)


#Same data progressing of testing data
test_data = test_data.drop(columns=['is_free'])
test_data['purchase_date'] = pd.to_datetime(test_data['purchase_date']).apply(lambda x: x.date())
test_data['release_date'] = pd.to_datetime(test_data['release_date']).apply(lambda x: x.date())
test_data['diff'] = test_data['purchase_date'] - test_data['release_date']
test_data['pmonth'] = test_data['purchase_date'].apply(lambda dt: dt.month)
test_data['pday'] = test_data['purchase_date'].apply(lambda dt: dt.day)
test_data['rmonth'] = test_data['release_date'].apply(lambda dt: dt.month)
test_data['rday'] = test_data['release_date'].apply(lambda dt: dt.day)
test_data.fillna(0, inplace=True)
test_data['diff'] = test_data['diff'].apply(lambda x: x.total_seconds()/86400)
test_data['rate'] = test_data['total_positive_reviews'] / test_data['total_negative_reviews']
test_data['rate'] = test_data['rate'].apply(lambda x: 0 if(x == np.inf) else x)
test_data['previews_rate'] = test_data['total_positive_reviews'] / (test_data['total_positive_reviews'] + test_data['total_negative_reviews'])
test_data['nreviews_rate'] = test_data['total_negative_reviews'] / (test_data['total_positive_reviews'] + test_data['total_negative_reviews'])
test_data = test_data.drop(columns=['purchase_date', 'release_date'])
test_data = test_data.drop(columns=['genres', 'categories', 'tags'])
mean2 = test_data.mean()
test_data.fillna(mean2, inplace=True)



print(train_data.info())
train_data.to_csv('s.csv', index=False)

x_train = train_data.drop(columns='playtime_forever')
y_train = train_data['playtime_forever']
kfold = KFold(n_splits = 5)

#RandomForestRegressor
rf = RandomForestRegressor(n_estimators=5, random_state = 0)
np.mean(cross_val_score(rf, x_train, y_train, cv=kfold))
# Train the model on training data
rf.fit(x_train, y_train)
predictions = rf.predict(test_data)

result = pd.DataFrame()
result['id'] = test_data['id']
result['playtime_forever'] = predictions
result['playtime_forever'] = result['playtime_forever'].apply(lambda x : 0 if(x<0) else x)

#print(result.tail(30))
result.to_csv('submission2.csv', index=False)


