#An Weigang
# catboost

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

train_data = train_data.drop(columns=['is_free'])
train_data['purchase_date'] = pd.to_datetime(train_data['purchase_date']).apply(lambda x: x.date())
train_data['release_date'] = pd.to_datetime(train_data['release_date']).apply(lambda x: x.date())
train_data['diff'] = train_data['purchase_date'] - train_data['release_date']
train_data.fillna(0, inplace=True)
train_data['diff'] = train_data['diff'].apply(lambda x: x.total_seconds()/86400)
train_data['rate'] = train_data['total_positive_reviews'] / train_data['total_negative_reviews']
train_data['rate'] = train_data['rate'].apply(lambda x: 0 if(x == np.inf) else x)
train_data = train_data.drop(columns=['purchase_date', 'release_date'])
# train_data['genres'] = train_data['genres'].str.split(',')
# train_data['genres'] = train_data['genres'].apply(lambda x: len(x))
# train_data['categories'] = train_data['categories'].str.split(',')
# train_data['categories'] = train_data['categories'].apply(lambda x: len(x))
# train_data['tags'] = train_data['tags'].str.split(',')
# train_data['tags'] = train_data['tags'].apply(lambda x: len(x))
# train_data['tnum'] = train_data['genres']+train_data['categories']+train_data['tags']
train_data = train_data.drop(columns=['genres','categories', 'tags'])
mean1 = train_data.mean()
train_data.fillna(mean1, inplace=True)
#train_data['playtime_forever'] = train_data['playtime_forever'].apply(lambda x: (x-train_data['playtime_forever'].min())/(train_data['playtime_forever'].max()-train_data['playtime_forever'].min()))
train_data['price'] = train_data['price'].apply(lambda x: (x-train_data['price'].min())/(train_data['price'].max()-train_data['price'].min()))
train_data['total_positive_reviews'] = train_data['total_positive_reviews'].apply(lambda x: (x-train_data['total_positive_reviews'].min())/(train_data['total_positive_reviews'].max()-train_data['total_positive_reviews'].min()))
train_data['total_negative_reviews'] = train_data['total_negative_reviews'].apply(lambda x: (x-train_data['total_negative_reviews'].min())/(train_data['total_negative_reviews'].max()-train_data['total_negative_reviews'].min()))
train_data['diff'] = train_data['diff'].apply(lambda x: (x-train_data['diff'].min())/(train_data['diff'].max()-train_data['diff'].min()))
train_data['rate'] = train_data['rate'].apply(lambda x: (x-train_data['rate'].min())/(train_data['rate'].max()-train_data['rate'].min()))


test_data = test_data.drop(columns=['is_free'])
test_data['purchase_date'] = pd.to_datetime(test_data['purchase_date']).apply(lambda x: x.date())
test_data['release_date'] = pd.to_datetime(test_data['release_date']).apply(lambda x: x.date())
test_data['diff'] = test_data['purchase_date'] - test_data['release_date']
test_data.fillna(0, inplace=True)
test_data['diff'] = test_data['diff'].apply(lambda x: x.total_seconds()/86400)
test_data = test_data.drop(columns=['purchase_date', 'release_date'])
test_data['rate'] = test_data['total_positive_reviews'] / test_data['total_negative_reviews']
test_data['rate'] = test_data['rate'].apply(lambda x: 0 if(x == np.inf) else x)
# test_data['genres'] = test_data['genres'].str.split(',')
# test_data['genres'] = test_data['genres'].apply(lambda x: len(x))
# test_data['categories'] = test_data['categories'].str.split(',')
# test_data['categories'] = test_data['categories'].apply(lambda x: len(x))
# test_data['tags'] = test_data['tags'].str.split(',')
# test_data['tags'] = test_data['tags'].apply(lambda x: len(x))
# test_data['tnum'] = test_data['genres']+test_data['categories']+test_data['tags']
test_data = test_data.drop(columns=['genres', 'categories', 'tags'])
mean2 = test_data.mean()
test_data.fillna(mean2, inplace=True)
# test_data['playtime_forever'] = test_data['playtime_forever'].apply(lambda x: int(x))
test_data['price'] = test_data['price'].apply(lambda x: (x-test_data['price'].min())/(test_data['price'].max()-test_data['price'].min()))
test_data['total_positive_reviews'] = test_data['total_positive_reviews'].apply(lambda x: (x-test_data['total_positive_reviews'].min())/(test_data['total_positive_reviews'].max()-test_data['total_positive_reviews'].min()))
test_data['total_negative_reviews'] = test_data['total_negative_reviews'].apply(lambda x: (x-test_data['total_negative_reviews'].min())/(test_data['total_negative_reviews'].max()-test_data['total_negative_reviews'].min()))
test_data['diff'] = test_data['diff'].apply(lambda x: (x-test_data['diff'].min())/(test_data['diff'].max()-test_data['diff'].min()))
test_data['rate'] = test_data['rate'].apply(lambda x: (x-test_data['rate'].min())/(test_data['rate'].max()-test_data['rate'].min()))



print(train_data.info())
train_data.to_csv('sl.csv', index=False)

x_train = train_data.drop(columns='playtime_forever')
y_train = train_data['playtime_forever']
logreg = LinearRegression()
logreg.fit(x_train, y_train)

#print(y_train.astype(int))

y_pred = logreg.predict(test_data)

result = pd.DataFrame()
result['id'] = test_data['id']
result['playtime_forever'] = y_pred
result['playtime_forever'] = result['playtime_forever'].apply(lambda x : -x if(x<0) else x)
#result['playtime_forever'] = result['playtime_forever'].apply(lambda x : round(x))
#print(result.tail(30))
result.to_csv('submission4.csv', index=False)









