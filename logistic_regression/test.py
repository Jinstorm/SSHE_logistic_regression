# list1 = ["这", "是", "一个", "测试"]
# list2 = ["这", "是", "一个", "测试"]
# for index, item in enumerate(list1), enumerate(list2):
#     print(index, item)
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDRegressor
import numpy as np
# a = np.array([7,8,9])
# v_r = a.reshape(1,-1)   #将数组a转化为行向量
# x = np
# print(v_r)

# a = np.array([[1],[2],[3]])
# x = np.log(a)
# print(x.shape)
# print(2*x)


# a = np.array([[1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01, 3.001e-01,
#   1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01, 8.589e+00, 1.534e+02,
#   6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02, 3.003e-02, 6.193e-03, 2.538e+01,
#   1.733e+01, 1.846e+02, 2.019e+03, 1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01,
#   4.601e-01, 1.189e-01]])

# weight = np.ones(30).reshape(1,-1)
# print(weight)
# y_predict = 0.25 * np.dot(a, weight.T)+0.5
# print("y_predict: ",0.25 * np.dot(a, weight.T)+0.5)
# selfwx = np.dot(a, weight.T)
# y = np.array([[0.]])
# error = y - y_predict
# g = np.dot(error.T, a)
# print("gradient: ", g)

# ## - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
# wx = -0.5 * selfwx
# ywx = selfwx * y
# print("ywx: ", ywx)
# wx_square = selfwx * selfwx * -0.125



# def read_data():
#     # from sklearn.datasets import load_breast_cancer
#     from sklearn.model_selection import train_test_split
#     from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
#     mm = MinMaxScaler()
#     ss = StandardScaler()

#     # cancers = load_breast_cancer()
#     # X = cancers.data       #获取特征值
#     # Y = cancers.target     #获取标签
#     # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/logistic_regression/X1.txt", X, delimiter=',') #, fmt='%.2f')
#     # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/logistic_regression/Y1.txt", Y, delimiter=',') #, fmt='%.2f')
#     # print("saving...")
#     X = np.loadtxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/logistic_regression/X.txt", delimiter=',') #, dtype = float)
#     Y = np.loadtxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/logistic_regression/Y.txt", delimiter=',') #, dtype = float)
#     # X = normalize(X,'l2')
#     X = ss.fit_transform(X)
#     print(X.shape)         #查看特征形状
#     print(Y.shape)         #查看标签形状
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = False)
#     return x_train, y_train, x_test, y_test

# X_data, y_data, X_test, y_test = read_data()
# print("::::::::::X::::::::::::\n", X_data[0:1,])
# np.random.shuffle(X_data)
# print("::::::::::X::::::::::::\n", X_data[0:1,])


# a = np.array((2,3,4,5))
# b = np.array((1,2,3,4))
# print(a*b)
# print((a+b)*(a+b))
# print(a*a+2*a*b+b*b)

 
c = np.array([[1,2,0],[4,0,1],[0,5,0]])
c[c!=0] = -1
print(c)
i = 1
c = 2
# raise NotImplementedError("Invalid rate.")
while i <= 10:
    print(i)
    if c == 1 or i == 10:
        print("hello")
    i = i +1

