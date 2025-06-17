# import pandas as pd
# train = pd.read_csv('train490.csv')
# test = pd.read_csv('test490.csv')
#
# print(train.head())
# print(test.head())
# print(train.info())
# print(test.info())
# print(train.describe())
# print(train.describe(include='object'))
# print(train.describe(include=object))
# print(test.describe(include=object))
#
# print(train.isnull().sum())
# print(test.isna().sum())
#
# print(train.shape,test.shape)
# target = train.pop('Segmentation')
# print(train.shape,test.shape)
# train=pd.get_dummies(train)
# test=pd.get_dummies(test)
# print(train.shape,test.shape)
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(train,
#                                                  target,
#                                                  test_size=0.2,
#                                                  random_state=0)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.metrics import f1_score
#
# lr=LogisticRegression(random_state=0)
# lr.fit(X_train,y_train)
# pred=lr.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# dt=DecisionTreeClassifier(random_state=0)
# dt.fit(X_train,y_train)
# pred=dt.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# rf=RandomForestClassifier(random_state=0)
# rf.fit(X_train,y_train)
# pred=rf.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
# lg.fit(X_train,y_train)
# pred=lg.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# xg=xgb.XGBClassifier(random_state=0)
# xg.fit(X_train,y_train)
# pred=xg.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# pred=lg.predict(test)
# submit=pd.DataFrame({
#     'ID':test['ID'],
#     'Segmentaion':pred
# })
#
# submit.to_csv('result1.csv',index=False)
#
# res=pd.read_csv('result1.csv')
# print(res)
# from sklearn.metrics import f1_score
# import pandas as pd
# import lightgbm as lgb
# from sklearn.metrics import f1_score
# train=pd.read_csv('train490.csv')
# test=pd.read_csv('test490.csv')
# target=train.pop('Segmentation')
#
# id제외
# train.drop('ID',axis=1,inplace=True)
# test_ID=test.pop('ID')
#
# train=pd.get_dummies(train)
# test=pd.get_dummies(test)
#
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(train,
#                                                  target,
#                                                  test_size=0.2,
#                                                  random_state=0)
# print(X_train,X_test.shape,y_train.shape,y_test.shape)
# lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
# lg.fit(X_train,y_train)
# pred=lg.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# import pandas as pd
# train=pd.read_csv('train490.csv')
# test=pd.read_csv('test490.csv')
# target=train.pop('Segmentation')
#
# from sklearn.preprocessing import LabelEncoder
# print(train.info())
##cols=train.select_dtypes(include=['object']).columns
# cols=['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']
#
# for col in cols:
#     le=LabelEncoder()
#     train[col]=le.fit_transform(train[col])
#     test[col]=le.transform(test[col])
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(train,
#                                                  target,
#                                                  test_size=0.2,
#                                                  random_state=0)
#
# lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
# lg.fit(X_train,y_train)
# pred=lg.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# import pandas as pd
# train=pd.read_csv('train490.csv')
# test=pd.read_csv('test490.csv')
# target = train.pop('Segmentation')
#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# cols = ['ID','Age','Work_Experience','Family_Size']
# train[cols]=scaler.fit_transform(train[cols])
# test[cols]=scaler.transform(test[cols])
#
# train=pd.get_dummies(train)
# test=pd.get_dummies(test)
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(train,
#                                                  target,
#                                                  test_size=0.2,
#                                                  random_state=0)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#
# lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
# lg.fit(X_train,y_train)
# pred=lg.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# pred=lg.predict(test)
# submit=pd.DataFrame({'ID':test['ID'],'Segmentation':pred})
# submit.to_csv('result2.csv',index=False)
#
# import pandas as pd
# train =pd.read_csv('train490.csv')
# test = pd.read_csv('test490.csv')
# print(train.head())
# print(test.head())
# print(train.info())
# print(test.info())
# print(train.describe())
# print(test.describe())
# print(train.describe(include=object))
# print(test.describe(include=object))
# print(train.shape,test.shape)
#
# target = train.pop('Segmentation')
# train = pd.get_dummies(train)
# test = pd.get_dummies(test)
# print(train.shape,test.shape)
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(train,
# target,
# test_size=0.2,
# random_state=42)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
# import lightgbm as lgb
# from sklearn.metrics import f1_score
#
# lr=LogisticRegression(random_state=0)
# lr.fit(X_train,y_train)
# pred=lr.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# dt=DecisionTreeClassifier(random_state=0)
# dt.fit(X_train,y_train)
# pred=dt.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# rf=RandomForestClassifier(random_state=0)
# rf.fit(X_train,y_train)
# pred=rf.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
#xg=xgb.XGBClassifier(random_state=0)
#xg.fit(X_train,y_train)
#pred=predict(X_test)
#print(f1_score(y_test,pred,average='macro'))
#
# lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
# lg.fit(X_train,y_train)
# pred=lg.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
#pred=lg.predict(test)
#submit=pd.DataFrame({
#    "ID":test['ID'],
#    'Segmentation': pred
#})
#submit.to_csv('submission.csv,index=False')
#
# import pandas as pd
# train = pd.read_csv('train490.csv')
# test =pd.read_csv('test490.csv')
# print(train.shape,test.shape)
# target=train.pop('Segmentation')
# print(train.info())
# from sklearn.preprocessing import LabelEncoder
# cols=['Gender','Ever_Married','Graduated','Profession','Spending_Score','Var_1']
# for col in cols:
#     le=LabelEncoder()
#     train[col]=le.fit_transform(train[col])
#     test[col]=le.transform(test[col])
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.2,random_state=0)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisiontreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#import lightgbm as lgb
#
# lg=lgb.LGBMClassifier(random_state=0,verbose=-1)
# lg.fit(X_train,y_train)
# pred=lg.predict(X_test)
# print(f1_score(y_test,pred,average='macro'))
#
# pred=lg.predict(test)
# submit=pd.DataFrame({'ID':test['ID'],'Segmentation':pred})
# submit.to_csv('submission.csv',index=False)

import pandas as pd
train =pd.read_csv('train505.csv')
test=pd.read_csv('test505.csv')
print(train.head())
print(test.head())
print(train.describe())
print(test.describe())
print(train.info())
print(test.info())
print(train.describe(include=object))
print(test.describe(include=object))
print(train.isnull().sum())
print(test.isnull().sum())
print(train.shape,test.shape)
target=train.pop('price')
train=pd.get_dummies(train)
test=pd.get_dummies(test)
print(train.shape,test.shape)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.2,random_state=0)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error


dt=DecisionTreeRegressor(random_state=0)
dt.fit(X_train,y_train)
pred = dt.predict(X_test)
print(root_mean_squared_error(y_test,pred))

rf=RandomForestRegressor(random_state=0)
rf.fit(X_train,y_train)
pred=rf.predict(X_test)
print(root_mean_squared_error(y_test,pred))

lg=lgb.LGBMRegressor(random_state=0,verbose=-1)
lg.fit(X_train,y_train)
pred=lg.predict(X_test)
print(root_mean_squared_error(y_test,pred))

#pred=dt.predict(test)
#result=pd.DataFrame({'pred':pred})
#result.to_csv('result.csv',index=False)

import pandas as pd
train=pd.read_csv('train505.csv')
test=pd.read_csv('test505.csv')
target=train.pop('price')

from sklearn.preprocessing import LabelEncoder
cols = ['model','transmission','fuelType',]
for col in cols:
    le = LabelEncoder()
    train[col]=le.fit_transform(train[col])
    test[col]=le.transform(test[col])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,target,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
print(root_mean_squared_error(y_test,pred))

import lightgbm
lg=lgb.LGBMRegressor(random_state=0,verbose=-1)
lg.fit(X_train,y_train)
pred=lg.predict(X_test)
print(root_mean_squared_error(y_test,pred))

pred=lg.predict(test)
result=pd.DataFrame({'pred':pred})
result.to_csv("result.csv",inplace=False)
