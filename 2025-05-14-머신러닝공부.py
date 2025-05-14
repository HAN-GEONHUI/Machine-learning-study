import pandas as pd
titanic_df = pd.read_csv('titanic_train.csv')
print(titanic_df.info())

print('단일 칼럼 데이터 추출:\n',titanic_df['Pclass'].head(3))
print('\n여러 칼럼의 데이터 추출:\n',titanic_df[['Survived','Pclass']].head(3))
#print('[] 안에 숫자 index는 KeyError 오류 발생:\n',titanic_df[0]) ->#데이터 프레임의 []내에 숫자 값을 입력할 경우 오류가 발생함.

print(titanic_df[0:2])
print(titanic_df[titanic_df['Pclass']==3].head(3))

data = {'Name':['Chulmin','Eunkyung','Jinwoong','Male'],
        'Year':[2011,2016,2015,2015],
        'Gender':['Male','Female','Male','Male']}
data_df= pd.DataFrame(data,index=['one','two','three','four'])
print(data_df)

print(data_df.iloc[0,0])

#아래 코드는 오류를 발생시킵니다.
#print(date_df.iloc[0,'Name'])
#print(data_df.iloc['one',0])
print('\n')
print('\n',data_df.iloc[1,0])
print('\n',data_df.iloc[2,1])
print('\n',data_df.iloc[0:2,[0,1]])
print('\n',data_df.iloc[0:2,0:3])
print('\n',data_df.iloc[:])
print('\n',data_df.iloc[:,:])

print("\n 맨 마지막 칼럼 데이터[:,-1]\n",data_df.iloc[:,-1])
print("\n 맨 마지막 칼럼을 제외한 모든 데이터 [:,:-1]\n",data_df.iloc[:,:-1])

print(data_df.loc['one','Name'])

#다은코드는 오류를 발생시킵니다
#print(data_df.loc[0,'Name'])

print('위치기반 iloc slicing\n',data_df.iloc[0:1,0],'\n')
print('명칭기반 loc slicing\n', data_df.loc['one':'two','Name'],'\n')
print(data_df.loc['three','Name'],'\n')
print(data_df.loc['one':'two',['Name','Year']],'\n')
print(data_df.loc['one':'three','Name':'Gender'],'\n')
print(data_df.loc[:],'\n')
print(data_df.loc[data_df.Year >=2014],'\n')

#참고
#iloc[ ] 뒤에는 숫자만 와야 하고,loc[ ] 는 주로 문자가 오지만 숫자도 올 수 있다.
#->loc[ 0:1,0] 에서 행은 1까지가 아니고 0이고,loc[one:two,0]에서는 행은 one,two까지이다.

#불린 인덱싱

titanic_df = pd.read_csv('titanic_train.csv')
titanic_boolean = titanic_df[titanic_df['Age']>60]
print(type(titanic_boolean))
print(titanic_boolean)

print(titanic_df[titanic_df['Age']>60][['Name','Survived']].head(3))
print(titanic_df.loc[titanic_df['Age']>60,['Name','Survived']].head(3))


# 1.and 조건일 때는 &
# 2.or 조건일 때는 |
# 3.Not 조건일 때는 ~
print('\n',titanic_df[(titanic_df['Age']>60)&(titanic_df['Pclass']==1)&
                 (titanic_df['Sex']=='female')])
cond1=titanic_df['Age']>60
cond2=titanic_df['Pclass']==1
cond3=titanic_df['Sex']=='female'
print('\n',titanic_df[cond1 & cond2 & cond3])

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',15)

#정렬- sort_values()
titanic_sorted = titanic_df.sort_values(by=['Name'])
print(titanic_sorted.head(3))
titanic_sorted = titanic_df.sort_values(by=['Pclass','Name'],ascending=False)
print(titanic_sorted.head(3))

#Aggrregation 함수 적용
print(titanic_df.count())

print(titanic_df[['Age','Fare']].mean())

#groupby()적용

titanic_groupby = titanic_df.groupby(by = 'Pclass')
print(type(titanic_groupby))

titanic_groupby = titanic_df.groupby('Pclass').count()
print(titanic_groupby)
titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId','Survived']].count()
print(titanic_groupby)

#titanic_df.groupby('Pclass')['Age'].agg([max,min])
# ->FutureWarning은 현재 코드는 작동하지만, 앞으로의 pandas 버전에서 동작 방식이 바뀔 예정이므로 수정이 필요하다는 경고가 뜸 그래서 책에서 나오는 방식말고 다른 방식을 사용해야 함.
print(titanic_df.groupby('Pclass')['Age'].agg(['max','min']))

agg_format = {'Age':'max','SibSp':'sum','Fare':'mean'}
print(titanic_df.groupby('Pclass').agg(agg_format))