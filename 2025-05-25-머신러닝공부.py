import sklearn
from pandas.core.internals.construction import dataclasses_to_dicts
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

#붓꽃 데이터 세트를 로딩합니다
iris = load_iris()

#iris.data는 Iris 데이터 세트에서 피처(feature)만으로 된 데이터를 numpy로 가지고 있습니다.
iris_data = iris.data

#iris.target은 붓꽃 데이터 세트에서 레이블(결정 값) 데이터를 numpy로 가지고 있습니다.
iris_label = iris.target
print('iris target값:',iris_label)
print('iris target명:',iris.target_names)

#붓꽃 데이터 세트를 자세히 보기 위해 DataFrame으로 변환합니다
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)
iris_df['label']=iris.target
print(iris_df.head(3))

X_train,X_test,y_train,y_test = train_test_split(iris_data,iris_label,test_size=0.2,random_state=11)

#DecisionTreeClassifier 객체 생성
dt_clf = DecisionTreeClassifier(random_state=11)

#학습 수행
dt_clf.fit(X_train,y_train)

#학습이 완료된 DecisionTreeClassfier 객체에서 테스트 데이터 세트로 예측 수행
pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print('예측 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))

from sklearn.datasets import load_iris

iris_data = load_iris()
print(type(iris_data))

keys = iris_data.keys()
print('붓꽃 데이터 세트의 키들:',keys)

print('\n feature_names의 type:',type(iris_data.feature_names))
print('feature_names 의 shape:',len(iris_data.feature_names))
print(iris_data.feature_names)

print('\n target_names 의 type:',type(iris_data.target_names))
print('target_names의 shape:',len(iris_data.target_names))
print(iris_data.target_names)

print('\n data 의 type:',type(iris_data))
print(' data의 shape:',iris_data.data.shape)
print(iris_data['data'])

print('\n target 의 type:',type(iris_data.target))
print(' target 의 shape:', iris_data.target.shape)
print(iris_data.target)

#학습/테스트 데이터 세트 분리 - train_test_split()
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data,train_label)

#학습 데이터 세트으로 예측 수행
pred = dt_clf.predict(train_data)
print('예측 정확도:',accuracy_score(train_label,pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

df_clf = DecisionTreeClassifier()
iris_data = load_iris()

X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,\
                                                 test_size=0.3,random_state=121)

dt_clf.fit(X_train,y_train)
pred = dt_clf.predict(X_test)
print('예측 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))

#K 폴드 교차 검증
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

# 5개의 폴드 세트로 분리하는 KFold 객체와 폴드 세트별 정확도를 담은 리스트 객체 생성.
kfold = KFold(n_splits=5)
cv_accuracy = []
print('붓꽃 데이터 세트 크기:',features.shape[0])

n_iter = 0

#KFold 객체의 split()릃 호출하면 폴드별,학습용,검증용 테스트의 로우 인덱스를 array로 반환
for train_index, test_index in kfold.split(features):
    # kfold.split() 으로 반환된 인덱스를 이용해 학습용,검증용 테스트 데이터 추출
    X_train,X_test = features[train_index],features[test_index]
    y_train,y_test = label[train_index],label[test_index]
    #학습 및 예측
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)
    n_iter +=1
    #반복 시마다 정확도 측정
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데잍터 크기: {2}, 검증 데이터 크기:{3}'
          .format(n_iter,accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)

# 개별 iteration별 정확도를 합하여 평균 정확도 계산
print('\n## 평균 검증 정확도:',np.mean(cv_accuracy))

#Stratified K 폴드
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
iris_df['label']=iris.target
print(iris_df['label'].value_counts())

kfold = KFold(n_splits=3)
n_iter = 0
for train_index, test_index in kfold.split(iris_df):
    n_iter +=1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증: {0}'.format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3)
n_iter=0

for train_index,test_index in skf.split(iris_df,iris_df['label']):
    n_iter +=1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증:{0}',format(n_iter))
    print('학습 레이블 데이터 분포:\n',label_train.value_counts())
    print('검증 레이블 데이터 분포:\n',label_test.value_counts())

dt_clf = DecisionTreeClassifier(random_state=156)

skfold = StratifiedKFold(n_splits=3)
n_iter = 0
cv_accuracy = []

#StratifiedKFold의 split() 호출시 반드시 레이블 데이터 세트도 추가 입력 필요
for train_index,test_index in skfold.split(features,label):
    #split()으로 반환된 인덱스를 이용해 학습용,검증용 테스트 데이터 추출
    X_train, X_test= features[train_index],features[test_index]
    y_train, y_test = label[train_index],label[test_index]

    #학습 및 예측
    dt_clf.fit(X_train,y_train)
    pred = dt_clf.predict(X_test)

    #반복 시마다 정확도 측정
    n_iter +=1
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 : {1}, 학습 데이터 크기: {2}, 검증 데이터 크기:{3}'
          .format(n_iter,accuracy,train_size,test_size))
    print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,test_index))
    cv_accuracy.append(accuracy)

#교차 검증별 정확도 및 평균 정확도 계산
print('\n## 교차 검증별 정확도:',np.round(cv_accuracy,4))
print('\n## 평균 검증 정확도:',np.round(np.mean(cv_accuracy),4))