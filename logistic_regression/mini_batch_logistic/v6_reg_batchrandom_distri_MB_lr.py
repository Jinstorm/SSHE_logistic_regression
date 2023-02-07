'''
# 分布式小批量梯度下降逻辑回归实现 mini-batch logistic regression 2022.11.29

# 在前序"小批量梯度下降逻辑回归"实现基础上, 提供了数据集纵向划分的分布式处理能力
# 分布式设置: 模型实例化时, 使用参数ratio规定拥有label一方所拥有的特征占总特征的比例
        内部对于分片的特征数量下取整;
        forward和backward过程分别更新权重;
        计算error时需要合并预测值;
        最终重构模型需要将两者的权重向量拼接.
2022.12.2
# 增加了mini-batch的batch打乱的功能, 更有效地防止过拟合
# 增加并完善了read自己data的函数, 注意数据集中标签值是1,-1还是1,0
# 准备实验spice

'''
import numpy as np
import math
import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

class LogisticRegression:
    """
    logistic回归
    """
    def __init__(self, weight_vector, batch_size, max_iter, alpha, eps, ratio = None, penalty = None, lambda_para = 1):
        """
        构造函数:初始化
        """
        self.model_weights = weight_vector
        self.batch_size = batch_size # 设置的batch大小
        self.batch_num = [] # 存储每个batch的大小
        self.n_iteration = 0
        self.max_iter = max_iter
        self.alpha = alpha
        self.pre_loss = 0
        self.eps = eps # 训练的误差下限
        self.ratio = ratio # 数据集划分比例
        self.penalty = penalty # 正则化策略
        self.lambda_para = lambda_para # 正则化系数


    def _cal_z(self, weights, features, party = None):
        # print("cal_z party: ", party)
        # print(features.shape, weights.shape)
        if party == "A": self.wx_self_A = np.dot(features, weights.T)
        elif party == "B": self.wx_self_B = np.dot(features, weights.T)
        else: self.wx_self = np.dot(features, weights.T)
        # return self.wx_self

    def _compute_sigmoid(self, z):
        # return 1 / (1 + np.exp(-z))
        return z * 0.25 + 0.5 

    def _compute_loss_cross_entropy(self, weights, label, batch_idx):
        """
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
        """
        # print("type label: ", type(label), type(self.wx_self), label.shape, self.wx_self.shape)
        half_wx = -0.5 * self.wx_self
        ywx = self.wx_self * label
        # ywx = np.multiply(self.wx_self, label)

        wx_square = self.wx_self * self.wx_self * -0.125 # wx_square = np.dot(self.wx_self.T, self.wx_self) * -0.125 # 这里后续要修改，两方平方，有交叉项
        # wx_square = np.multiply(self.wx_self, self.wx_self) * -0.125
        batch_num = self.batch_num[batch_idx]

        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )
        return loss

    def distributed_compute_loss_cross_entropy(self, label, batch_num):
        """
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
        """
        self.encrypted_wx = self.wx_self_A + self.wx_self_B
        half_wx = -0.5 * self.encrypted_wx
        ywx = self.encrypted_wx * label

        wx_square = (2*self.wx_self_A * self.wx_self_B + self.wx_self_A * self.wx_self_A + self.wx_self_B * self.wx_self_B) * -0.125 # wx_square = np.dot(self.wx_self.T, self.wx_self) * -0.125 # 这里后续要修改，两方平方，有交叉项
        # batch_num = self.batch_num[batch_idx]

        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )
        return loss

    def _compute_loss(self, y, label, batch_idx):
        batch_num = self.batch_num[batch_idx]
        loss = -1 * label * np.log(y) - (1 - label) * np.log(1 - y)
        return np.sum(loss)


    def forward(self, weights, features): #, batch_weight):
        self._cal_z(weights, features, party = None)
        # sigmoid
        sigmoid_z = self._compute_sigmoid(self.wx_self)
        return sigmoid_z

    def distributed_forward(self, weights, features, party = None): #, batch_weight):
        # print("party: ", party)
        self._cal_z(weights, features, party)
        # sigmoid
        if party == "A":
            sigmoid_z = self._compute_sigmoid(self.wx_self_A)
        elif party == "B":
            sigmoid_z = self._compute_sigmoid(self.wx_self_B)
        return sigmoid_z

    def backward(self, error, features, batch_idx):
        # print("batch_idx: ",batch_idx)
        batch_num = self.batch_num[batch_idx]
        gradient = np.dot(error.T, features) / batch_num
        return gradient

    def distributed_backward(self, error, features, batch_num):
        # print("batch_idx: ",batch_idx)
        # batch_num = self.batch_num[batch_idx]
        gradient = np.dot(error.T, features) / batch_num
        return gradient

    def check_converge_by_loss(self, loss):
        converge_flag = False
        if self.pre_loss is None:
            pass
        elif abs(self.pre_loss - loss) < self.eps:
            converge_flag = True
        self.pre_loss = loss
        return converge_flag
    
    def shuffle_data(self, Xdatalist, Ydatalist):
        # X_batch_list
        # np.random.shuffle(X_data)
        zip_list = list( zip(Xdatalist, Ydatalist) )              # 将a,b整体作为一个zip,每个元素一一对应后打乱
        np.random.shuffle(zip_list)               # 打乱c
        Xdatalist[:], Ydatalist[:] = zip(*zip_list)
        return Xdatalist, Ydatalist
    
    def shuffle_distributed_data(self, XdatalistA, XdatalistB, Ydatalist):
        # X_batch_list
        # np.random.shuffle(X_data)
        zip_list = list( zip(XdatalistA, XdatalistB, Ydatalist) )              # 将a,b整体作为一个zip,每个元素一一对应后打乱
        np.random.shuffle(zip_list)               # 打乱c
        XdatalistA[:], XdatalistB[:], Ydatalist[:] = zip(*zip_list)
        return XdatalistA, XdatalistB, Ydatalist

    def fit_model(self, X_train, Y_train, instances_count):
        # mini-batch 数据集处理
        print("ratio: ",self.ratio)
        if self.ratio is None:
            X_batch_list, y_batch_list = self._generate_batch_data(X_train, Y_train, self.batch_size)
        else: 
            X_batch_listA, X_batch_listB, y_batch_list = self._distributed_generate_batch_data(X_train, Y_train, self.batch_size, self.ratio)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分

        self.n_iteration = 0
        self.loss_history = []
        test = 0
        
        while self.n_iteration < self.max_iter:
            loss_list = []
            batch_labels = None
            if self.ratio == None:
                # 打乱数据集的batch
                X_batch_list, y_batch_list = self.shuffle_data(X_batch_list, y_batch_list)

                for batch_idx, batch_data in enumerate(X_batch_list):
                    batch_labels = y_batch_list[batch_idx].reshape(-1, 1) # 转换成1列的列向量
                    
                    ## forward and backward
                    y = self.forward(self.model_weights, batch_data)
                    error = y - batch_labels # 注意这里谁减谁，和后面更新梯度是加还是减有关
                    gradient = self.backward(error = error, features = batch_data, batch_idx = batch_idx)

                    ## compute loss
                    batch_loss = self._compute_loss_cross_entropy(weights = self.model_weights, label = batch_labels, batch_idx = batch_idx)
                    # batch_loss = self._compute_loss(y, batch_labels, batch_idx)
                    loss_list.append(batch_loss)

                    ## update model
                    if self.penalty == 'l2':
                        batch_num = self.batch_num[batch_idx]
                        self.model_weights = self.model_weights - self.alpha * gradient - self.lambda_para * self.alpha * self.model_weights / batch_num
                    elif self.penalty == 'l1':
                        batch_num = self.batch_num[batch_idx]
                        self.model_weights = self.model_weights - self.alpha * gradient - self.lambda_para * self.alpha * np.sign(self.model_weights) / batch_num
                    else: self.model_weights = self.model_weights - self.alpha * gradient     # 30*1
                    # self.model_weights = self.model_weights - self.alpha * gradient     # 30*1
            
            # distributed
            else:
                # 打乱数据集的batch
                X_batch_listA, X_batch_listB, y_batch_list = self.shuffle_distributed_data(X_batch_listA, 
                                    X_batch_listB, y_batch_list)
                # if self.n_iteration == 0: 
                #     self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
                # print("weightA, weightB: ", self.weightA.shape, self.weightB.shape)
                for batch_dataA, batch_dataB, batch_labels, batch_num in zip(X_batch_listA, X_batch_listB, y_batch_list, self.batch_num):
                    batch_labels = batch_labels.reshape(-1, 1)

                    ## forward and backward
                    y1 = self.distributed_forward(self.weightA, batch_dataA, party = "A")
                    y2 = self.distributed_forward(self.weightB, batch_dataB, party = "B")
                    error = (y1 + y2) - batch_labels # 注意这里谁减谁，和后面更新梯度是加还是减有关
                    # print("error: ", error)
                    self.gradient1 = self.distributed_backward(error = error, features = batch_dataA, batch_num = batch_num)
                    self.gradient2 = self.distributed_backward(error = error, features = batch_dataB, batch_num = batch_num)
                    
                    ## compute loss
                    batch_loss = self.distributed_compute_loss_cross_entropy(label = batch_labels, batch_num = batch_num)
                    loss_list.append(batch_loss)

                    ## update model
                    # self.model_weights = self.model_weights - self.alpha * gradient     # 30*1
                    self.weightA = self.weightA - self.alpha * self.gradient1
                    self.weightB = self.weightB - self.alpha * self.gradient2
                    # print("weightA, B: ", self.weightA.shape, self.weightB.shape, self.weightA[0][0], self.weightB[0][0])
            # print("gradientA, B: ", self.gradient1.shape, self.gradient1.shape, self.gradient1[0][0], self.gradient2[0][0])
            # print("weightA, B: ", self.weightA.shape, self.weightB.shape, self.weightA[0][0], self.weightB[0][0])
            ## 计算 sum loss
            loss = np.sum(loss_list) / instances_count
            print("\rIteration {}, batch sum loss: {}".format(self.n_iteration, loss))
            self.loss_history.append(loss)

            ## 判断是否停止
            self.is_converged = self.check_converge_by_loss(loss)
            if self.is_converged:
                # self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
                if self.ratio is not None: 
                    self.model_weights = np.hstack((self.weightA, self.weightB))
                    print(self.model_weights)
                break

            self.n_iteration += 1

    def _generate_batch_data(self, X, y, batch_size):
        '''
        生成mini-batch数据集
        '''
        X_batch_list = []
        y_batch_list = []
        
        for i in range(len(y) // batch_size):
            X_batch_list.append(X[i * batch_size : i * batch_size + batch_size, :])
            y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)
        
        if (len(y) % batch_size > 0):
            X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)

        return X_batch_list, y_batch_list

    def _distributed_generate_batch_data(self, X, y, batch_size, ratio):
        '''
        生成两部分,纵向划分的, mini-batch数据集.
        只划分特征矩阵,不划分标签向量y
        ratio: 决定划分到含有标签的一方的数据的比例,对划分的数量下取整,例如 0.8 * 63 -> 50
        '''
        X_batch_listA = []
        X_batch_listB = []
        y_batch_list = []

        self.indice = math.floor(ratio * X.shape[1]) # 纵向划分数据集，位于label一侧的特征数量
        
        for i in range(len(y) // batch_size):
            X_tmpA, X_tmpB = np.hsplit(X[i * batch_size : i * batch_size + batch_size, :], [self.indice]) # 纵向划分数据集
            X_batch_listA.append(X_tmpA)
            X_batch_listB.append(X_tmpB)
            y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)
        # ------------------------------------------ 修改到这里了
        if (len(y) % batch_size > 0):
            X_tmpA, X_tmpB = np.hsplit(X[len(y) // batch_size * batch_size:, :], [self.indice])
            # X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            X_batch_listA.append(X_tmpA)
            X_batch_listB.append(X_tmpB)
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)

        return X_batch_listA, X_batch_listB, y_batch_list # listA——持有label一侧，较多样本; listB——无label一侧



    def predict(self, x_test, y_test):
        z = np.dot(x_test, self.model_weights.T)
        y = self._compute_sigmoid(z)
        score = 0
        for i in range(len(y)):
            if y[i] >= 0.5: y[i] = 1
            else: y[i] = 0
            if y[i] == y_test[i]:
                score += 1
            else:
                pass
        rate = score/len(y)
        print("Predict precision: ", rate)

def read_data():
    # from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
    mm = MinMaxScaler()
    ss = StandardScaler()

    # cancers = load_breast_cancer()
    # X = cancers.data       #获取特征值
    # Y = cancers.target     #获取标签
    # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/logistic_regression/X1.txt", X, delimiter=',') #, fmt='%.2f')
    # np.savetxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/logistic_regression/Y1.txt", Y, delimiter=',') #, fmt='%.2f')
    # print("saving...")
    X = np.loadtxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/logistic_regression/X.txt", delimiter=',') #, dtype = float)
    Y = np.loadtxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/logistic_regression/Y.txt", delimiter=',') #, dtype = float)
    # X = normalize(X,'l2')
    X = ss.fit_transform(X)
    print(X.shape)         #查看特征形状
    print(Y.shape)         #查看标签形状
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle = False)
    return x_train, y_train, x_test, y_test

def read_sampling_data():
    from sklearn.datasets import load_svmlight_file
    import os
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
    mm = MinMaxScaler()
    ss = StandardScaler()

    dataset_file_name = 'splice'  
    train_file_name = 'splice_train.txt' 
    test_file_name = 'splice_test'
    main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'

    # dataset_file_name = 'a6a'
    # train_file_name = 'a6a.txt'
    # test_file_name = 'a6a.t'
    # main_path = '/Users/zbz/data/'
    train_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, train_file_name))
    test_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, test_file_name))
    X_train = train_data[0]
    Y_train = train_data[1].astype(int)
    X_test = test_data[0]
    Y_test = test_data[1].astype(int)
    # print(X_train) # 1000 * 60
    # print(Y_train[0]) # 1000 * 1

    # 判断标签是(1;-1)还是 (1;0), 将-1的标签全部转化成0标签
    if -1 in Y_train:  
        Y_train[Y_train == -1] = 0
        Y_test[Y_test == -1] = 0
    print(Y_train)
    print(Y_test)

    # #a6a a7a
    # X_train = X_train.todense().A
    # X_train = np.hstack( (X_train, np.zeros(X_train.shape[0]).reshape(-1, 1)) )
    # return ss.fit_transform(X_train), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array

    # #splice
    return ss.fit_transform(X_train.todense().A), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array
    # return X_train.todense().A, Y_train, X_test.todense().A, Y_test # matrix转array
    
    # X = np.loadtxt('/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/X_train_sketch.txt', 
    #                 delimiter=',') #, dtype = float)
    

if __name__ == "__main__":
    # X_data, y_data, X_test, y_test = read_data()
    X_data, y_data, X_test, y_test = read_sampling_data()
    print(X_data.shape, X_data.shape[0], X_data.shape[1], y_data.shape, X_test.shape, y_test.shape)
    np.random.seed(100)
    # # 正态
    # weight_vector = np.random.normal(0.0, 0.0001, X_data.shape[1]).reshape(1, -1)
    # # 全1
    # weight_vector = np.ones(X_data.shape[1]).reshape(1,-1)
    # # 随机
    # weight_vector = np.random.random(X_data.shape[1]).reshape(1, -1)
    # weight = (0.00001-0.0) * weight_vector + 0.0
    # # 全0
    weight_vector = np.zeros(X_data.shape[1]).reshape(1, -1)

    # print(weight_vector)
    # 模型实例化
    # LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
    #                 max_iter = 1000, alpha = 0.0001, eps = 1e-6, ratio = 0.7)  # 0.956140350877193
                    # splice 分布式: Predict precision:  0.8496551724137931
    LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
                    max_iter = 2000, alpha = 0.0001, eps = 1e-6, penalty = None, lambda_para = 1)  
                    # breast_cancer: 0.9649122807017544
                    # splice: 0.8482758620689655


    # 训练
    LogisticRegressionModel.fit_model(X_data, y_data, X_data.shape[0])

    # plt.plot(LogisticRegressionModel.loss_history)
    # plt.show()
    LogisticRegressionModel.predict(X_test, y_test)

    # dis: 0.7719298245614035