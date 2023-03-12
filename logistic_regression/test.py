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

 
# c = np.array([[1,2,3],[2,3,4]])
# c = np.array([[1,2,3],[2,3,4]])
# c = c.shape
# # print(c.shape)
# print("shape:{}".format(c[0]))


def test():
    import argparse
    # 创建解析步骤
    parser = argparse.ArgumentParser(description='ababa')

    # 添加参数步骤
    parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers')
    # 解析参数步骤  
    args = parser.parse_args()
    print(args.accumulate(args.integers))

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="annotated data set for aligned")
    parser.add_argument('-r1', '--src-data-root', dest='data_root1', required=True, default='/mnt/dms_data/Data_All/custom_alarm_smokeall_20190505/json/v20190610', type=str, metavar='STRING', help='src data path')
    parser.add_argument('-r2', '--dst-data-root', dest='data_root2', required=True, default='/mnt/dms_data/Data_All/custom_alarm_smokeall_20190505/json/v20190505', type=str, metavar='STRING', help='tag data path')
    parser.add_argument('-r3', '-a', dest='data_root3', required=True, type=str, metavar='STRING', choices=['raw', 'sketch'], help='tag data path')
    args = parser.parse_args()
    data_root1 = args.data_root1
    data_root2 = args.data_root2
    print('data_root1 path is %s' % data_root1)  # data_root1 path is /mnt/dms_data/Data_All/custom_alarm_smokeall_20190505/json/v20190610
    print('data_root2 path is %s' % data_root2)  # data_root1 path is /mnt/dms_data/Data_All/custom_alarm_smokeall_20190505/json/v20190610