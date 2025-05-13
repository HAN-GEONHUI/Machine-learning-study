import pandas as pd
titanic_df = pd.read_csv('titanic_train.csv')
print(titanic_df)
print(titanic_df.head(3))
print('DataFrame 크기:',titanic_df.shape)
titanic_df.info()
print(titanic_df.describe())
value_counts=titanic_df['Pclass'].value_counts()
print(value_counts)
titanic_pclass=titanic_df['Pclass']
print(type(titanic_pclass))
print(titanic_pclass.head())
value_counts = titanic_df['Pclass'].value_counts()
print(type(value_counts))
print(value_counts)
print('titanic_df 데이터 건수:',titanic_df.shape[0])
print('기본 설정인 dropna = True로 value_counts()')
print(titanic_df['Embarked'].value_counts())
print(titanic_df['Embarked'].value_counts(dropna=False))

import numpy as np
col_name1=['col1']
list1 = [1,2,3]
array1 = np.array(list1)
print('array1 shape:',array1.shape)
df_list1=pd.DataFrame(list1,columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n',df_list1)
df_array1=pd.DataFrame(array1,columns=col_name1)
print('1차원 리스트로 만든 DataFrame:\n',df_array1)

col_name2=['col1','col2','col3']

list2=[[1,2,3],
       [11,12,13]]
array2 = np.array(list2)
print('array2 shape:',array2.shape)
df_list2 = pd.DataFrame(list2,columns=col_name2)
print('2차원 리스트로 만든 DataFrame:\n',df_list2)
df_array2=pd.DataFrame(array2,columns=col_name2)
print('2차원 ndarray로 만든 DataFrame:\n',df_array2)

dict={'col1':[1,11],'col2':[2,22],'col3':[3,33]}
df_dict = pd.DataFrame(dict)
print('딕셔너리로 만든 DataFrame:\n',df_dict)

#DataFrame을 ndarray로 변환
array3 =df_dict.values
print('df_dict.values 타입:',type(array3),'df_dict.values shape:',array3.shape)
print(array3)

list3=df_dict.values.tolist()
print('df_dict.values.totlist()타입:',type(list3))
print(list3)

dict3 = df_dict.to_dict('list')
print('\n df_dict.to_dict()타입:',type(dict3))
print(dict3)

titanic_df['Age_0']=0
print(titanic_df.head(3))

titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_NO'] = titanic_df['SibSp'] + titanic_df['Parch']+1
print(titanic_df.head(3))

titanic_df['Age_by_10'] = titanic_df['Age_by_10']+100
print(titanic_df.head(3))

titanic_drop_df = titanic_df.drop('Age_0',axis=1)
print(titanic_df.head(3))

# 모든 열을 출력하도록 설정
pd.set_option('display.max_columns', None)

titanic_drop_df = titanic_df.drop('Age_0',axis=1)
print(titanic_df.head(3))

drop_result = titanic_df.drop(['Age_0','Age_by_10','Family_NO'],axis=1,inplace=False)
print('inplace =True 로 drop 후 반환된 값:',drop_result)
print(titanic_drop_df.head(3))

drop_result = titanic_df.drop(['Age_0','Age_by_10','Family_NO'],axis=1,inplace=True)
print('inplace =True 로 drop 후 반환된 값:',drop_result)
print(titanic_df.head(3))

#drop_result = titanic_df.drop(['Age_0','Age_by_10','Family_NO'],axis=1,inplace=False)
#요렇게 저장한다고 할때 drop_result에는 컬럼 'Age_0','Age_by_10','Family_NO'를 뺀 나머지 컬럼들이 저장되고 titanic_df에는 변함이 없음

#반대로
#drop_result = titanic_df.drop(['Age_0','Age_by_10','Family_NO'],axis=1,inplace=True)
#이 상황에서 drop_result에 저장되는거는 아무것도 없는 'None'이 뜬다. 그리고 titanic_df에도 컬럼 'Age_0','Age_by_10','Family_NO'이 삭제되어 있는다. inplace=True 조심해서 쓸것

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',15)
print('### before axis 0 drop ####')
print(titanic_df.head(3))

titanic_df.drop([0,1,2],axis=0,inplace=True)

print('### after axis 0 drop ####')
print(titanic_df.head(3))

#원본 파일 다시 로딩
titanic_df = pd.read_csv('titanic_train.csv')
#인덱스 객체 추출
indexes = titanic_df.index
print(indexes)
#Index 객체를 실제 값 array로 변환
print('Index 객체 array값:\n',indexes.values)

print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])

#indexes[0] = 5 -> 요거는 못바꿈(한 번 만들어진 DataFrame 및 Series의 index 객체는 함부로 변경할 수 없다.

series_fair = titanic_df['Fare']
print('Fair Series max 값:',series_fair.max())
print('Fair Series sum 값:',series_fair.sum())
print('sum()Fair Series:',sum(series_fair))
print('Fair Series + 3:\n',(series_fair + 3).head(3))

titanic_reset_df = titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)

print('### before reset_index ###')
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)
print('value_counts 객체 변수 타입:',type(value_counts))
new_value_counts = value_counts.reset_index(inplace=False)
print('### After reset_index ###')
print(new_value_counts)
print('new_value_counts 객체 변수 타입:',type(new_value_counts))

