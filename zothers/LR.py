import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from collections import Counter

def sigmoid(x):
    x = np.array(x)
    return 1. / (1. + np.exp(-x))

## 目标函数, 极大似然
## 注意这里求取了平均值而不是直接 sum
def L(w, b, X, y):
    dot = np.dot(X, w) + b
    return np.mean(y * dot - np.log(1 + np.exp(dot)), axis=0)

## w, b 的导数
def dL(w, b, X, y):
    dot = np.dot(X, w) + b
    distance = y - sigmoid(dot)
    distance = distance.reshape(-1, 1)
    return np.mean(distance * X, axis=0), np.mean(distance, axis=0)

## 随机梯度下降? (上升)
def sgd(w, b, X, y, epoch, lr):
    for i in range(epoch):
        dw, db = dL(w, b, X, y)
        w += lr * dw
        b += lr * db
    return w, b

## 测试代码, 对于预测值, 当概率大于 0.5 时, label 属于 True
def predict(w, b, X_test):
    return sigmoid(np.dot(X_test, w) + b) >= 0.5

## 画出分类面
def plot_surface(X, y, w, b):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    X_test = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(w, b, X_test)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    counter = Counter(y)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    ## 画出分割线
    #     i = np.linspace(x_min, x_max, 100)
    #     o = (w[0] * i + b) / -w[1]
    #     ax.plot(i, o)
    
    for label in counter.keys():
        ax.scatter(X[y==label, 0], X[y==label, 1])
    plt.show()

# if 
## 训练代码
iris = load_iris()
X = iris.data[:100, :2]
y = iris.target[:100] # y \in {0, 1}
feature_names = iris.feature_names[2:]
np.random.seed(123)
n = X.shape[1]
# print(X.shape) # 100*2 100个样本两个特征
w = np.random.randn(n)
b = np.random.randn(1)
print('initial: w: {}, b: {}, L: {}'.format(w, b, L(w, b, X, y)))
w, b = sgd(w, b, X, y, 10000, 0.001)
print('final: w: {}, b: {}, L: {}'.format(w, b, L(w, b, X, y)))

plot_surface(X, y, w, b)
