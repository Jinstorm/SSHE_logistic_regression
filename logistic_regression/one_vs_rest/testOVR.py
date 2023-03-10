import numpy as np
a = []
c = 1
def zerodef(arr):
    print(c)
    return np.where(arr == 3, 1, 0)
b = np.array([1,2,3,4,5])
a.append(b)
a.append(b)
a.append(b)

a = list(map(zerodef, a))

print(a)


def test_ovr_():
    prob_lst = []          
    for i in range(0,7):
        a = np.array([i-2,i-1,i,i+1])
        a = np.where(a > 0, a, 0)#.reshape(1,-1)
        print(a)
        print(a.tolist())
        print(np.shape(a.tolist()))
        prob_lst.append(a.tolist())
    print("LOOP over.")
    print(prob_lst)

    prob_array = np.asarray(prob_lst)
    # print(prob_array)
    print(prob_array.shape)
    y_predict = []    
    label_lst = [0,1,2,3]       
    for i in range(0,7):
        temp = list(prob_array[i])
        # print(temp)
        index = temp.index(max(temp))
        # print(index)
        y_predict.append(label_lst[index])
    print("++++++++++++")
    print(y_predict)


# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# #使用pandas库读取函数
# def per_read(path):
#     data=pd.read_csv(path) #pandas读取数据
#     class_dict={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2} #字典转换
#     data['species']=data['species'].map(class_dict) #使用map进行替换
#     #print(data)
#     #进行数据分割
#     x=data.iloc[:,0:-1]
#     y=data.iloc[:,-1]
#     x=np.array(x,dtype=np.float)
#     y = np.array(y, dtype=np.float)
#     #数据归一化处理
#     mu=x.mean(0)
#     std=x.std(0)
#     x=(x-mu)/std
#     #分测试集与数据集
#     x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
#     return x_train,x_test,y_train,y_test

# #OVO的计算函数
# def ovo(x_train,x_test,y_train,y_test):
#     log_model=LogisticRegression(solver="lbfgs",multi_class="multinomial")
#     ##self.solver参数决定了我们对逻辑回归损失函数的优化方法,lbfgs是拟牛顿法multi_class="multinomial"是制定使用ovo
#     log_model.fit(x_train,y_train)
#     y_predict=log_model.predict(x_test)
#     print(np.sum(y_predict==y_test)/len(y_test)) #计算预测的准确率

# #OVR的计算函数
# def ovr(x_train,x_test,y_train,y_test):
#     log_model=LogisticRegression(solver="lbfgs",multi_class="ovr")
#     #是用牛顿法进行优化，并指定使用ovr
#     log_model.fit(x_train,y_train)
#     y_predict=log_model.predict(x_test)
#     print(np.sum(y_predict==y_test)/len(y_test))

# if __name__=="__main__":
#     x_train, x_test, y_train, y_test =per_read("F:\comdata\iris.txt")
#     ovo(x_train, x_test, y_train, y_test)
#     ovr(x_train, x_test, y_train, y_test)
