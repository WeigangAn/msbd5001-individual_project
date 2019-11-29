#MSBD 5001 individual project
#An Weigang
#model2 - LinearRegression

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
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import datetime
import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv('train.csv', low_memory=False)
test_data =  pd.read_csv('test.csv', low_memory=False)

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
train_data['diff'] = train_data['diff'].apply(lambda x: x.total_seconds()/86400)
train_data['rate'] = train_data['total_positive_reviews'] / train_data['total_negative_reviews']
train_data['rate'] = train_data['rate'].apply(lambda x: 0 if(x == np.inf) else x)
train_data = train_data.drop(columns=['purchase_date', 'release_date'])
train_data = train_data.drop(columns=['genres','categories', 'tags'])
mean1 = train_data.mean()
train_data.fillna(mean1, inplace=True)
# min-max scale
train_data['price'] = train_data['price'].apply(lambda x: (x-train_data['price'].min())/(train_data['price'].max()-train_data['price'].min()))
train_data['total_positive_reviews'] = train_data['total_positive_reviews'].apply(lambda x: (x-train_data['total_positive_reviews'].min())/(train_data['total_positive_reviews'].max()-train_data['total_positive_reviews'].min()))
train_data['total_negative_reviews'] = train_data['total_negative_reviews'].apply(lambda x: (x-train_data['total_negative_reviews'].min())/(train_data['total_negative_reviews'].max()-train_data['total_negative_reviews'].min()))
train_data['diff'] = train_data['diff'].apply(lambda x: (x-train_data['diff'].min())/(train_data['diff'].max()-train_data['diff'].min()))
train_data['rate'] = train_data['rate'].apply(lambda x: (x-train_data['rate'].min())/(train_data['rate'].max()-train_data['rate'].min()))
train_data['pmonth'] = train_data['pmonth'].apply(lambda x: (x-train_data['pmonth'].min())/(train_data['pmonth'].max()-train_data['pmonth'].min()))
train_data['pday'] = train_data['pday'].apply(lambda x: (x-train_data['pday'].min())/(train_data['pday'].max()-train_data['pday'].min()))
train_data['rmonth'] = train_data['rmonth'].apply(lambda x: (x-train_data['rmonth'].min())/(train_data['rmonth'].max()-train_data['rmonth'].min()))
train_data['rday'] = train_data['rday'].apply(lambda x: (x-train_data['rday'].min())/(train_data['rday'].max()-train_data['rday'].min()))

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
test_data = test_data.drop(columns=['purchase_date', 'release_date'])
test_data['rate'] = test_data['total_positive_reviews'] / test_data['total_negative_reviews']
test_data['rate'] = test_data['rate'].apply(lambda x: 0 if(x == np.inf) else x)
test_data = test_data.drop(columns=['genres', 'categories', 'tags'])
mean2 = test_data.mean()
test_data.fillna(mean2, inplace=True)
# min-max scale
test_data['price'] = test_data['price'].apply(lambda x: (x-test_data['price'].min())/(test_data['price'].max()-test_data['price'].min()))
test_data['total_positive_reviews'] = test_data['total_positive_reviews'].apply(lambda x: (x-test_data['total_positive_reviews'].min())/(test_data['total_positive_reviews'].max()-test_data['total_positive_reviews'].min()))
test_data['total_negative_reviews'] = test_data['total_negative_reviews'].apply(lambda x: (x-test_data['total_negative_reviews'].min())/(test_data['total_negative_reviews'].max()-test_data['total_negative_reviews'].min()))
test_data['diff'] = test_data['diff'].apply(lambda x: (x-test_data['diff'].min())/(test_data['diff'].max()-test_data['diff'].min()))
test_data['rate'] = test_data['rate'].apply(lambda x: (x-test_data['rate'].min())/(test_data['rate'].max()-test_data['rate'].min()))
test_data['pmonth'] = test_data['pmonth'].apply(lambda x: (x-test_data['pmonth'].min())/(test_data['pmonth'].max()-test_data['pmonth'].min()))
test_data['pday'] = test_data['pday'].apply(lambda x: (x-test_data['pday'].min())/(test_data['pday'].max()-test_data['pday'].min()))
test_data['rmonth'] = test_data['rmonth'].apply(lambda x: (x-test_data['rmonth'].min())/(test_data['rmonth'].max()-test_data['rmonth'].min()))
test_data['rday'] = test_data['rday'].apply(lambda x: (x-test_data['rday'].min())/(test_data['rday'].max()-test_data['rday'].min()))


print(train_data.info())
#train_data.to_csv('sl.csv', index=False)

#LinearRegression
x_train = train_data.drop(columns='playtime_forever')
y_train = train_data['playtime_forever']
logreg = LinearRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(test_data)

result = pd.DataFrame()
result['id'] = test_data['id']
result['playtime_forever'] = y_pred
result['playtime_forever'] = result['playtime_forever'].apply(lambda x : -x if(x<0) else x)
#result['playtime_forever'] = result['playtime_forever'].apply(lambda x : round(x))
#print(result.tail(30))
result.to_csv('submission4.csv', index=False)









