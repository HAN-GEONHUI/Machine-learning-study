# import pandas as pd
# import numpy as np
# ten=pd.read_csv('members.csv')
# print(ten.head())
# print(ten.info())
# tensort=ten.sort_values(by='views',ascending=False)
# print(tensort)
# ten10= tensort['views'].iloc[:10]
# print(ten10)
# tenth = tensort['views'].iloc[9]
# print(tenth)
# tensort['views'].iloc[:10]=tenth
# print(tensort)
# men=tensort[tensort['age']>=80]['views'].mean()
# print(round(men,2))

# import pandas as pd
# import numpy as np
# eight = pd.read_csv('members.csv')
# eighteen = eight.iloc[:int(len(eight)*0.8),:]
# print(eighteen)
# before=eighteen['f1'].std()
# print(before)
# print(eighteen['f1'].isna().sum())
# print(eighteen['f1'].isnull().sum())
# after=eighteen['f1'].fillna(eighteen['f1'].median())
# print(after)
# after8=after.std()
# print(after8)
# print(round(abs(before-after8)),2)

# import pandas as pd
# df=pd.read_csv('members.csv')
# std=df['age'].std()
# mean=df['age'].mean()
#
# lower=mean-(std*1.5)
# upper=mean+(std*1.5)
#
# cond1=df['age']<lower
# cond2=df['age']>upper
#
# print(df[cond1|cond2]['age'].sum())

# import pandas as pd
# import numpy as np
# df=pd.read_csv('members.csv')
# print(df.info())
# print(df.isna().sum())
# a=df.dropna(axis=0)
# print(a.info())
# b=df.dropna(axis=1)
# print(b.info())
# seventeen=a.iloc[:int(len(a)*0.7),:]
# print(seventeen.info())
# print(int(seventeen['f1'].quantile(0.25)))
# print(int(seventeen['f1'].quantile(.25)))

# import pandas as pd
# df=pd.read_csv('year.csv')
# print(df.info())
# print(df)
# tweenty=pd.read_csv('year.csv',index_col=0)
# print(tweenty)
#
# row = tweenty.loc[2000]
# greater_than_mean = (row > row.mean()).sum()
# print(greater_than_mean)
# m=tweenty.loc[2000].mean()
# print(sum(tweenty.loc[2000,:]>m))

# import pandas as pd
# import numpy as np
# df=pd.read_csv('members.csv')
# print(df.isna().sum())
# df_cntNull = df.sort_values(by=df.isna().sum(),ascending=False)
# print(df_cntNull)

# import pandas as pd
# import numpy as np
# df = pd.read_csv('data4-1.csv')
# print(df.info())
# first= df['age'].quantile(0.25)
# third= df['age'].quantile(0.75)
# print(first)
# print(third)
# result=abs(first-third)
# print(int(result))

# import pandas as pd
# import numpy as np
# df=pd.read_csv('data4-2.csv')
# print(df.info())
# print(df)
# cond1=(df['loves']+df['wows'])/df['reactions']>0.4
# cond2=(df['loves']+df['wows'])/df['reactions']<0.5
# cond3=df['type']=='video'
# print(len(df[cond1&cond2&cond3]))

#import pandas as pd
# df=pd.read_csv('data4-3.csv')
# print(df)
# print(df.info())
# print(df['date_added'].head())
# cond1 = df['date_added'] == 'January'
# cond2 = df['date_added'] == '2018'
# cond3 = df['country'] == 'United Kingdom'
# print(len(df[cond1&cond2&cond3]))

# import pandas as pd
# import numpy as np
# df=pd.read_csv('data4-3.csv')
# df['date_added']=pd.to_datetime(df['date_added'])
#
# df['year']=df['date_added'].dt.year
# df['month']=df['date_added'].dt.month
#
# print(df.info())
# cond1 = df['country'] == 'United Kingdom'
# cond2 = df['year'] == 2018 # = df['date_added']>= '2018-1-1'
# cond3 = df['month'] == 1 # = df['date_added']<= '2018-1-31'
#
# print(len(df[cond1 & cond2 &cond3]))

# import pandas as pd
# import numpy as np
# df=pd.read_csv('data5-1.csv')
# print(df.info())
# cond1 = df['종량제봉투종류'] == '규격봉투'
# cond2 = df['종량제봉투용도'] == '음식물쓰레기'
# cond3 = df['2ℓ가격'] != 0

import pandas as pd
X_test = pd.read_csv('X_test.csv')
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

df=pd.concat([X_train,y_train['Reached.on.Time_Y.N']],axis=1)

print(X_train.shape,y_train.shape,X_test.shape)
print(X_train.head())
print(X_train.info())
print(y_train.head())
print(X_train.info())
print(X_train.describe())
print(X_train.describe(include='O'))
print(X_test.describe(include='O'))
print(X_train.isnull().sum())
print(X_test.isnull().sum())
print(y_train['Reached.on.Time_Y.N'].value_counts())
from sklearn.preprocessing import LabelEncoder
cols = X_train.select_dtypes(include='object').columns
#cols=['Warehouse_block','Mode_of_Shipment','Product_purchases','Gender']

import pandas as pd
from IPython.display import display  # 이 줄을 추가하세요

# 예시
display(X_train[cols].head(3))

for col in cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

print(X_train[cols].head(3))

X_train = X_train.drop('ID',axis=1)
X_test_id = X_test.pop('ID')

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(
    X_train,
    y_train['Reached.on.Time_Y.N'],
    test_size=0.2,
    random_state=0
)
print(X_tr.shape,X_val.shape,y_tr.shape,y_val.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

lr= LogisticRegression(random_state=0)
lr.fit(X_tr,y_tr)
pred = lr.predict_proba(X_val)
print(roc_auc_score(y_val,pred[:,1]))

dt=DecisionTreeClassifier(random_state=0)
dt.fit(X_tr,y_tr)
pred = dt.predict_proba(X_val)
print(roc_auc_score(y_val,pred[:,1]))

rf = RandomForestClassifier(random_state=0)
rf.fit(X_tr,y_tr)
pred = rf.predict_proba(X_val)
print(roc_auc_score(y_val,pred[:,1]))

xg = xgb.XGBClassifier(random_state=0)
xg.fit(X_tr,y_tr)
pred = xg.predict_proba(X_val)
print(roc_auc_score(y_val,pred[:,1]))

lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
lg.fit(X_tr,y_tr)
pred = lg.predict_proba(X_val)
print(roc_auc_score(y_val,pred[:,1]))

pred =lg.predict_proba(X_test)
print(pred)
submit = pd.DataFrame(
    {
        "ID":X_test_id,
        "Reached.on.Time_Y.N":pred[:,1]
    }
)
print(submit.head())
submit.to_csv('result.csv',index=False)
dd=pd.read_csv('result.csv')
print(dd)