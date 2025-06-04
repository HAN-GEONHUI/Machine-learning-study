from sklearn.datasets import load_iris
import pandas as pd
#붓꽃 데이터 세트를 로딩하고 DataFrame으로 변환합니다
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data,columns=iris.feature_names)

print('feature 들의 평균 값')
print(iris_df.mean())
print('\nfeature 들의 분산 값')
print(iris_df.var())

from sklearn.preprocessing import StandardScaler

#StandardScaler객체 생성
scaler = StandardScaler()
#StandardScaler로 데이터 세트 변환.fit()과 transform()호출.
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

#transform()시 스케일 변환된 데이터 세트가 NumPy ndarray로 반환돼 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
print('feature 들의 평균 값')
print(iris_df_scaled.mean())
print('\nfeature 들의 분산 값')
print(iris_df_scaled.var())

#MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

#MinMaxScaler객체 생성
scaler = MinMaxScaler()
#MinMaxScaler로 데이터 세트 변환.fit()과 transform() 호출.
scaler.fit(iris_df)

#transform()시 스케일 변형된 데이터 세트NumPy ndarray로 반환돼 이를 DataFrame으로 변환
iris_df_scaled = pd.DataFrame(data=iris_scaled,columns=iris.feature_names)
print('feature들의 최솟값')
print(iris_df_scaled.min())
print('\nfeature들의 최댓값')
print(iris_df_scaled.max())

#학습 데이터와 테스트 데이터의 스케일링 변환시 유의점
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#학습 데이터는 0부터 10까지,테스트 데이터는 0부터 5까지 값을 가지는 데이터 세트로 생성
#Scaler 클래스의 fit(),transform()2차원 이상 데이터만 가능하므로 reshape(-1,1)로 차원 변경
train_array = np.arange(0,11).reshape(-1,1)
test_array = np.arange(0,6).reshape(-1,1)

#MinMaxScaler 객체에 별도의 feature_range 파라미터 값을 지정하지 않으면 0-1 값으로 변환
scaler = MinMaxScaler()

#fit()하게 되면 train_array 데이터 최솟값이 0 ,최댓값이 10으로 설정.
scaler.fit(train_array)

#1/10 scale로 train_array 데이터 변환함.원본 10->1로 변환됨.
train_scaled = scaler.transform(train_array)

print('원본 train_array 데이터:',np.round(train_array.reshape(-1),2))
print('Scaled된 train_array 데이터:',np.round(train_scaled.reshape(-1),2))

#MinMaxScaler에 test_array를 fit()하게 되면 원본 데이터의 최솟값이 0,최댓값이 5로 설정됨.
scaler.fit(test_array)

#1/5 scale로 test_array 데이터 변환함. 원본 5->1로 변환.
test_scaled = scaler.transform(test_array)

#test_array의 scale변환 출력.
print('원본 test_array데이터:',np.round(test_array.reshape(-1),2))
print('Scale된 test_array 데이터:',np.round(test_scaled.reshape(-1),2))

scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
print('원본 train_array 데이터:',np.round(train_array.reshape(-1),2))
print('Scale된 train_array 데이터:',np.round(train_scaled.reshape(-1),2))

#test_array에 Scale 변환을 할 때에는 반드시 fit()을 호출하지 않고 transform()만으로 변환해야 함.
test_scaled = scaler.transform(test_array)
print('\n원본 test_array 데이터:',np.round(test_array.reshape(-1),2))
print('Sacle된 test_array 데이터:',np.round(test_scaled.reshape(-1),2))

#주의 해야 할점
#1.가능하다면 전체 데이터의 스케일링 변환을 적용한 뒤 학습과 테스트 데이터로 분리
#2.1이 여의치 않다면 테스트 데이터 변환 시에는 fit()이나 fit_transform()을 적용하지 않고 학습 데이터로 이미 fit()된 Scaler객체를 이용해 transform()으로 변환

#사이킷런으로 수행하는 타이타닉 생존자 예측
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.show()

titanic_df = pd.read_csv('./titanic_train.csv')
print(titanic_df.head())

print('\n ### 학습 데이터 정보 ### \n')
print(titanic_df.info())

#titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
#titanic_df['Cabin'].fillna('N',inplace=True)
#titanic_df['Embarked'].fillna('N',inplace=True)
#titanic_df['Age']는 실제 원본 DataFrame이 아니라 복사된 시리즈 객체일 가능성이 있어, inplace=True가 원래의 titanic_df에 반영되지 않을 수 있음을 경고가 나타남으로 안 쓰는게 남.

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('N')
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('N')
print('데이터 세트 Null값 개수',titanic_df.isnull().sum().sum())

print('Sex 값 분포:\n',titanic_df['Sex'].value_counts())
print('\n Cabin 값 분포:\n',titanic_df['Cabin'].value_counts())
print('\n Embarked 값 분포 : \n',titanic_df['Embarked'].value_counts())

titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))

print(titanic_df.groupby(['Sex','Survived'])['Survived'].count())

#1.print(titanic_df.groupby(['Sex','Survived'])['Survived'].count())
#2.print(titanic_df.groupby(['Sex','Survived']).count())
#둘이 비슷해 보이지만 차이가 있다.
#1->['Survived'] 컬럼 하나만 대상으로 각 그룹별로 개수(count)를 센다.
#2->그룹화 후 모든 컬럼에 대해 각 그룹의 결측치가 아닌 값의 개수(count)를 계산한다.

sns.barplot(x='Sex',y = 'Survived',data=titanic_df)
plt.show()  # <-- 그래프를 실제로 띄워주는 명령

sns.barplot(x='Pclass',y='Survived',hue='Sex',data=titanic_df)
plt.show()