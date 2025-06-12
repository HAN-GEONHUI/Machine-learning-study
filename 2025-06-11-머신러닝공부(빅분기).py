import pandas as pd
import numpy as np
file = pd.read_csv('P210201.csv')
print(file)
Age = file.sort_values(by='crim', ascending=False, inplace=False)
#Age = file['crim'].sort_values(ascending=False).head10 --> 요 방법도 있다.
print(Age)
A=Age['crim'].head(10)
print(A)
B=A.iloc[9]
print(B)
Age['crim']=np.where(Age['crim']>B,B,Age['crim'])
over80 = Age[Age['age']>=80]
print(over80['age'].describe())
print(round(over80['crim'].mean(),2))

import pandas as pd
import numpy as np
housing = pd.read_csv('P210202.csv')
print('housing이다',housing)
print(20639*0.8)
eight = housing.head(16511)

# 위에 방법처럼 임시로 하는 방법 말고 제대로 된 방법은 요거다.
#-> housing_80 = housing.iloc[:int(len(df) * 0.8),:]/슬라이싱만 수행합니다. housing의 원본 데이터를 참조하는 view(뷰) 객체를 만들 수 있습니다.
#이후 housing_80을 수정하면 원본 데이터에 영향을 줄 수 있습니다.

# housing_80 = housing.iloc[:int(len(df) * 0.8)].copy()/슬라이싱 후 복사본(copy) 을 생성합니다. housing_80은 완전히 독립된 객체입니다.이 경우에는 SettingWithCopyWarning도 피할 수 있고,
# 원본을 안전하게 보호할 수 있습니다.
print(eight)
before8=eight['total_bedrooms'].std()
print(before8)
print(eight['total_bedrooms'].isnull().sum())
print(eight['total_bedrooms'].isna().sum())
after=eight['total_bedrooms'].fillna(eight['total_bedrooms'].median())
print(after.isna().sum())
after8=after.std()
print(after8)
print(round((before8-after8),2))

housing08 = housing.iloc[:int(len(housing)*0.8),:]
housing08 = housing.iloc[:int(len(housing)*0.8)].copy()