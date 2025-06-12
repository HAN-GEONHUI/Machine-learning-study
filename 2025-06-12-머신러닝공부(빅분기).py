# import pandas as pd
# import numpy as np
# df=pd.read_csv('data4-3.csv')
# print(df.info())
# print(df['date_added'])
# df['date_added'] = pd.to_datetime(df['date_added'])
# print(df['date_added'])
# df['year'] = df['date_added'].dt.year
# df['month'] = df['date_added'].dt.month
# print(df['country'])
# cond1=df['year']==2018
# cond2=df['month']==1
##cond2=df['date_added']>='2018-1-1'
##cond3=df['date_added']<='2018-1-31'
# cond3=df['country'] == 'United Kingdom'
# print(len(df[cond1&cond2&cond3]))

# import pandas as pd
# train = pd.read_csv('train490.csv')
# test = pd.read_csv('test490.csv')
#
# print(train.shape,test.shape)
# print(train.head())
# print(test.head())
#
# print(train.describe())
# print(train.describe(include='object'))
# print(test.describe(include='object'))
#
# print(train.isnull().sum().sum())
# print(test.isnull().sum().sum())
#
# print(train['Segmentation'].value_counts())
#
# target = train.pop('Segmentation')
# print(train.shape,test.shape)
# train=pd.get_dummies(train)
# test=pd.get_dummies(test)
# print(train.shape,test.shape)
#
# from sklearn.model_selection import train_test_split
# X_tr,X_val,y_tr,y_val = train_test_split(train,
#                                          target,
#                                          test_size=0.2,
#                                          random_state=0)
# print(X_tr.shape,X_val.shape,y_tr.shape,y_val.shape)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.metrics import f1_score
#
# lr = LogisticRegression(random_state=0)
# lr.fit(X_tr,y_tr)
# pred=lr.predict(X_val)
# print(f1_score(y_val,pred,average='macro'))
#
# dt=DecisionTreeClassifier(random_state=0)
# dt.fit(X_tr,y_tr)
# pred=dt.predict(X_val)
# print(f1_score(y_val,pred,average='macro'))
#
# rf=RandomForestClassifier(random_state=0)
# rf.fit(X_tr,y_tr)
# pred=rf.predict(X_val)
# print(f1_score(y_val,pred,average='macro'))
#
#xg=xgb.XGBClassifier(random_state=0)
#xg.fit(X_tr,y_tr)
#pred=xg.predict(X_val)
#print(f1_score(y_val,pred,average='macro'))
#
# lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
# lg.fit(X_tr,y_tr)
# pred = lg.predict(X_val)
# print(f1_score(y_val,pred,average='macro'))
#
# pred = lg.predict(test)
# submit =pd.DataFrame({
#     'ID' : test['ID'],
#     'Segmentation' : pred
# })
# submit.to_csv("result.csv",index=False)
# ok=pd.read_csv('result.csv')
# print(ok)
#
# import pandas as pd
# train = pd.read_csv('train490.csv')
# test = pd.read_csv('test490.csv')
# target = train.pop('segmentation')
#
# train.drop('ID,axis=1',inplace=True)
# test_ID = test.pop('ID')
#
# train = pd.get_dummies(train)
# test = pd.get_dummies(test)
#
# from sklearn.model_selection import train_test_split
# X_tr,X_val,y_tr,y_val = train_test_split(train,
#                                          target,
#                                          test_size=0.2,
#                                          random_state=0)
# print(X_tr.shape,X_val.shape,y_tr.shape,y_val.shape)
#
# lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
# lg.fit(X_tr,y_tr)
# pred=lg.predict(X_val)
# print(f1_score(y_val,pred,average='macro'))
#

import pandas as pd
df = pd.read_csv('data5-1.csv')

print(df.info())
print(df.head())

cond1 = df['종량제봉투종류'] == '규격봉투'
cond2 = df['종량제봉투용도'] == '음식물쓰레기'
cond3 = df['2ℓ가격'] !=0
#print(len(df[cond1&cond2&cond3])) -> 요렇게 하면 틀림
df=df[cond1 & cond2 & cond3]

print(round(df['2ℓ가격'].mean()))

import pandas as pd
df = pd.read_csv('data5-2.csv')
print(df)
print(df.info())
print(df.head())

df['bmi'] = df['Weight']/(df['Height']/100)**2

cond1 = (df['bmi']>=18.5) & (df['bmi']<23)

cond2 = (df['bmi']>=23) &(df['bmi']<25)

print(abs(len(df[cond1])-len(df[cond2])))

import pandas as pd
df = pd.read_csv('data5-3.csv')
print(df)
print(df.info())
df['순전입학생'] = df['전입학생수(계)']-df['전출학생수(계)']
df.sort_values(by='순전입학생',ascending=False,inplace=True)
print(df.head())
print(int(df.iloc[0,-2]))