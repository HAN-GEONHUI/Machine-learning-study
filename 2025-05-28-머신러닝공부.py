#교차 검증을 보다 간편하게-cross_val_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris_data.data
label = iris_data.target

#성능 지표는 정확도(accuracy),교차 검증 세트는 3개
scores = cross_val_score(dt_clf,data,label,scoring='accuracy',cv=3)
print('교차 검증별 정확도:',np.round(scores,4))
print('평균 검증 정확도:',np.round(np.mean(scores),4))

#GridSearchCV - 교차 검증과 최적 하이퍼 파라미터 튜닝을 한 번에
grid_parameters = {'max_depth':[1,2,3],
                   'min_samples_split':[2,3]}

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#데이터를 로딩하고 학습 데이터와 테스트 데이터 분리
iris_data = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,
                                                 test_size=0.2,random_state=121)
dtree = DecisionTreeClassifier()

###파라미터를 딕셔너리 형태로 설정
parameters = {'max_depth':[1,2,3],'min_samples_split':[2,3]}

import pandas as pd

#param_grid의 하이퍼 피라미터를 3개의 train, test set fold로 나뉘어 테스트 수행 결정
### refit =True가 default임.True이면 가장 좋은 파라미터 설정으로 재학습시킴.
grid_dtree = GridSearchCV(dtree,param_grid=parameters,cv=3,refit=True)

#붓꽃 학습 데이터로 param_grid의 하이퍼 파라미터를 순차적으로 학습/평가
grid_dtree.fit(X_train,y_train)

# 모든 열을 출력하도록 설정
pd.set_option('display.max_columns', None)

#GridSearchCV 결과를 추출해 DataFrame으로 변환
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df[['params','mean_test_score','rank_test_score','split0_test_score','split1_test_score','split2_test_score']])

print('GridSearchCV 최적 피라미터:',grid_dtree.best_params_)
print('GridSearchCv 최고 정확도:{0:.4f}'.format(grid_dtree.best_score_))
# {0:.4f}-> 소수점 아래 4자리까지 실수(float)를 출력하겠다는 의미

#GridSearchCV의 refit으로 이미 학습된 estimator 반환
estimator = grid_dtree.best_estimator_

#GridSearchCV의 best_estimator_는 이미 최적 학습이 됐으므로 별도 학습이 필요 없음
pred = estimator.predict(X_test)
print('테스트 데이터 세트 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))

#레이블 인코딩
from sklearn.preprocessing import LabelEncoder

items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']

# LabelEncoder를 객체로 생선한 후, fit()과 transform()으로 레이블 인코딩 수행.
encoder = LabelEncoder() # LabelEncoder 객체 생성
encoder.fit(items) # 문자열 목록(items)을 학습하여 고유한 클래스(레이블) 목록을 만듦
labels = encoder.transform(items)  # 학습된 클래스를 기준으로 문자열을 숫자 레이블로 변환
print('인코딩 변환값:',labels) # 숫자로 변환된 결과 출력
print('인코딩 클래스:',encoder.classes_)
print('디코딩 원본값:',encoder.inverse_transform([4,5,2,0,1,1,3,3]))

# 선형회귀에서는 냉장고가:1, 믹서:2로 변환되면 1보다 2가 더 큰값이므로 ML알고리즘에서 가중치가 더 부여되거나 중요하게 인식 딜 가능성이 있으므로
# 레이블 인코딩은 선형회구 같은 ML알고리즘에는 사용하면 안되고 반대로 트리 계열의 ML 알고리즘은 숫자의 이러한 특성을 반영하지 않기 때문에 사용해도 된다.

# 원-핫 인코딩
# 원-핫 인코딩은 사이킷런에서 OneHotEncoder클래스로 변환이 가능하지만 입력값으로 2차원 데이터가 필요하고 OneHotEncoder를 이용해 변환한 값이 희소 행렬 형태(Sparse Matrix)이므로
# 이를 다시 toarray()메서드를 이용해 밀집행렬(Dense Matrix)로 변환해야 한다.

from sklearn.preprocessing import OneHotEncoder
import numpy as np

items = ['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']

#2차원 ndarray로 변환합니다.
items = np.array(items).reshape(-1,1)

#원-핫 인코딩을 적용합니다.
oh_encoder = OneHotEncoder()
oh_encoder.fit(items)
oh_labels =oh_encoder.transform(items)

#OneHotEncoder로 변환한 결과는 희소행렬이므로 toarray()를 이용해 밀집 행렬로 변환.
print('원-핫 인코딩 데이터')
print(oh_labels.toarray())
print('원-핫 인코딩 데이터 차원')
print(oh_labels.shape)

#희소행렬(Sparse Matrix)이란? 대부분의 값이 0인 행렬/데이터의 많은 부분이 "비어 있음(0)"/저장 공간을 아끼기 위해 0이 아닌 값만 저장하는 특별한 방식 사용
#위 행렬은 대부분이 0이기 때문에 "희소"하다고 표현한다.

#밀집행렬(Dense Matrix)이란? 대부분의 원소가 0이 아닌 값인 일반적인 행렬/우리가 일반적으로 생각하는 2차원 배열 구조/모든 값을 메모리에 전부 저장/숫자형 데이터를 다룰 때 주로 사용
#거의 모든 값이 0이 아니므로 "밀집"되어 있다고 표현한다.\

import pandas as pd

df = pd.DataFrame({'items':['TV','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서','믹서']})
print(pd.get_dummies(df))
#요즘은 결과가 0,1숫자로 표시되는게 아니라 True,False로 표시되는데  pd.get_dummies()의 출력 형식이 DataFrame 내부에서 Boolean 타입으로 설정된 경우라서 그런거다.
print(pd.get_dummies(df, dtype=int)) #요렇게 표시하면 0,1 숫자로 표시되어서 나온다.
