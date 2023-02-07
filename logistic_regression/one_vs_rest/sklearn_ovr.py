from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
iris = datasets.load_iris()
# print iris
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
'''sc.scale_标准差, sc.mean_平均值, sc.var_方差'''

lr = LogisticRegression(C=10000.0, random_state=0)
lr.fit(X_train_std, y_train)
print('系数：',lr.coef_,lr.intercept_)
# 预测
#print lr.intercept_
print ('phiz', 1.0/(1+np.e**(-(np.dot(lr.coef_, X_test_std[0])+lr.intercept_))))
print ('decision', lr.decision_function(X_test_std[0]))
print ('phiz',1.0/(1+np.e**(-lr.decision_function(X_test_std[0]))))
y_pred = lr.predict(X_test_std)
# print '测试集',X_test_std[0]
# print '预测值', y_pred[0]
# print '预测概率',lr.predict_proba(X_test_std[0])
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))