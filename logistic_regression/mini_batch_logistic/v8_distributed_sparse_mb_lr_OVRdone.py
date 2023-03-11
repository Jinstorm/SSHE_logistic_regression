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
2022.12.3
# 完成了数据集集中的sparse支持(通过初始化的时候选择data_tag的值-'sparse' or None), 目前分布式的还没有做sparse的支持
# 最新: 已经完成全部的集中式和分布式的稀疏矩阵的支持! 
         使用LIL格式在处理分片时初始化, 分片结束后每个batch转化为CSR矩阵加速运算(且这个转化操作是线性时间复杂度的)

# 先尝试增加一个读入encoded数据的read_data函数
# 修改了分布式训练中的两个错误:
    1. forward函数计算的时候, A、B两方分别计算sigmoid近似函数 0.25z+0.5, 再相加, 结果比正确结果多加了一个0.5
    2. 分布式初始化权重的时候, 意外将初始化的步骤写在了每个epoch的循环开头, 应该写在训练的最开始.

2022.12.4
# 之前的分布式不太对, 之前的是同一个矩阵划分成两个部分, 课题实验的场景是原来就是两个不同的横向对齐的数据特征矩阵
    因此, 应该在数据读入函数、batch数据生成函数、和 fit函数上作修改
    修改完成, 分布式kernel方法和集中式kernel方法的准确度相同
    这一版本打算: 删除之前的分布式数据读入和fit中的处理模块

2022.12.6
总结一下代码实现: 
目前集中式没有做sparse的支持(即定义模型的时候不加上ratio,会默认使用集中式数据训练), (其实可以支持, 得修改一下)
但是实现的伪分布式——即将一个整块数据划分成两部分数据训练, 其实效果和集中式是一样的, 
——即如果需要测试集中式的sparse可以使用伪分布式的例子, 输入的还是一个数据集, 随意设置ratio参数, 可以得到不同的数据划分的情况下的训练效果, 应该和集中式的准确度没有区别
这个"分布式"是支持sparse和dense输入的.

我们的论文中使用的方式: 使用这个训练函数: fit_model_distributed_input(), 且要设置ratio, 虽然用不到, 这个函数使用的是两部分sketch数据, 支持sparse和dense

【Todo】
见main函数: 这里的ratio在本课题中的纵向输入, 只会用到划分 model_weight, 但是由于两方数据集的数量不一定严格按照比例, 
容易导致weight比例和data比例不一致. 后续调整weight划分的依据.
# 纵向划分分布式(这里的ratio控制的是weight的划分比例, 需要根据输入的数据的划分比例手动确定)—— 其实ratio没有用, 容易和数据集的两部分数量不对应, 后面还要修改
    LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
                    max_iter = 2000, alpha = 0.0001, eps = 1e-6, "ratio = 0.7", penalty = None, lambda_para = 1, data_tag = None)

可以修改一下main中的训练接口, 现在有点乱
(这个版本是实现了: 纵向划分数据和集中输入数据的小批量梯度下降逻辑回归的版本小结)

'''
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
PATH_DATA = 'E:\\zbz\\code\\vscode_python\\hetero_sshe_logistic_regression\\data' # '../../data/'
class LogisticRegression:
    """
    logistic回归
    """
    def __init__(self, weight_vector, batch_size, max_iter, alpha, 
                        eps, ratio = None, penalty = None, lambda_para = 1, data_tag = None):
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
        self.data_tag = data_tag # 输入数据的格式 (目前需要支持两种格式: sparse和dense)


    def _cal_z(self, weights, features, party = None):
        # print("cal_z party: ", party)
        # print(features.shape, weights.shape)
        if party == "A":
            if self.data_tag == 'sparse': self.wx_self_A = features.dot(weights.T)
            else: self.wx_self_A = np.dot(features, weights.T)
        elif party == "B": 
            if self.data_tag == 'sparse': self.wx_self_B = features.dot(weights.T)
            else: self.wx_self_B = np.dot(features, weights.T)
            
        else: 
            if self.data_tag == 'sparse':
                self.wx_self = features.dot(weights.T)# np.dot(features, weights.T)
            elif self.data_tag == None:
                self.wx_self = np.dot(features, weights.T)
        # return self.wx_self

    def _compute_sigmoid(self, z):
        # return 1 / (1 + np.exp(-z))
        # print(type(z))
        # if self.data_tag == None: 
        return z * 0.25 + 0.5
        # elif self.data_tag == 'sparse': return z.todense() * 0.25 + 0.5

    def _compute_sigmoid_dual_distributed(self, z):
        # return 1 / (1 + np.exp(-z))
        # print(type(z))
        # if self.data_tag == None: 
        return z * 0.25

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
        # wx_square2 = self.encrypted_wx * self.encrypted_wx * -0.125
        # assert all(wx_square == wx_square2)  # 数组比较的返回值为: 类似[True False False]

        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )
        return loss

    # def _compute_loss(self, y, label, batch_idx):
    #     batch_num = self.batch_num[batch_idx]
    #     loss = -1 * label * np.log(y) - (1 - label) * np.log(1 - y)
    #     return np.sum(loss)


    def forward(self, weights, features): #, batch_weight):
        # print("weights: ", type(weights))
        # print("features: ", type(features))
        self._cal_z(weights, features, party = None)
        # sigmoid
        # print("self.wx_self: ", type(self.wx_self))
        # print(self.wx_self)
        sigmoid_z = self._compute_sigmoid(self.wx_self)
        return sigmoid_z

    def distributed_forward(self, weights, features, party = None): #, batch_weight):
        # print("party: ", party)
        self._cal_z(weights, features, party)
        # sigmoid
        if party == "A":
            sigmoid_z = self._compute_sigmoid(self.wx_self_A)
        elif party == "B":
            # sigmoid_z = self._compute_sigmoid(self.wx_self_B)
            # 注意这里考虑到分布式的计算问题, 少加了一个0.5, 这样后面y1+y2-label计算loss的时候才是正确的error值
            sigmoid_z = self._compute_sigmoid_dual_distributed(self.wx_self_B)
            
        return sigmoid_z

    def backward(self, error, features, batch_idx):
        # print("batch_idx: ",batch_idx)
        batch_num = self.batch_num[batch_idx]
        # print("error, feature shape: ", error.T.shape, features.shape)
        # print("error, feature type: ", type(error.T), type(features))
        if self.data_tag == 'sparse':
            gradient = features.T.dot(error).T / batch_num # 稀疏矩阵
        elif self.data_tag == None:
            gradient = np.dot(error.T, features) / batch_num # 非稀疏矩阵
        # print("gradient shape: ", gradient.shape)
        return gradient

    def distributed_backward(self, error, features, batch_num):
        # print("batch_idx: ",batch_idx)
        # batch_num = self.batch_num[batch_idx]
        if self.data_tag == 'sparse':
            gradient = features.T.dot(error).T / batch_num # 稀疏矩阵
        elif self.data_tag == None:
            gradient = np.dot(error.T, features) / batch_num # 非稀疏矩阵
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

    # 理想集中 和 将一部分数据划分为两部分的纵向分布式
    def fit_model(self, X_train, Y_train, instances_count):
        # mini-batch 数据集处理
        print("ratio: ", self.ratio)
        if self.ratio is None:
            X_batch_list, y_batch_list = self._generate_batch_data(X_train, Y_train, self.batch_size)
        elif self.data_tag == None: 
            X_batch_listA, X_batch_listB, y_batch_list = self._distributed_generate_batch_data(X_train, Y_train, self.batch_size, self.ratio)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
        elif self.data_tag == 'sparse':
            print('sprase data batch generating...')
            X_batch_listA, X_batch_listB, y_batch_list = self._distributed_generate_sparse_batch_data(X_train, Y_train, self.batch_size, self.ratio)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
            print('Generation done.')
        else:
            raise Exception("[fit model] No proper entry for batch data generation. Check the data_tag or the fit function.")
            
        self.n_iteration = 0
        self.loss_history = []
        test = 0
        
        while self.n_iteration < self.max_iter:
            loss_list = []
            batch_labels = None
            if self.ratio == None:
                
                for batch_idx, batch_data in enumerate(X_batch_list):
                    batch_labels = y_batch_list[batch_idx].reshape(-1, 1) # 转换成1列的列向量
                    # print("batch data: ", type(batch_data))
                    
                    ## forward and backward
                    y = self.forward(self.model_weights, batch_data)
                    error = y - batch_labels # 注意这里谁减谁，和后面更新梯度是加还是减有关
                    gradient = self.backward(error = error, features = batch_data, batch_idx = batch_idx)
                    ## compute loss
                    batch_loss = self._compute_loss_cross_entropy(weights = self.model_weights, label = batch_labels, batch_idx = batch_idx)
                    # batch_loss = self._compute_loss(y, batch_labels, batch_idx)
                    loss_list.append(batch_loss)
                    # print(error)
                    # print(gradient.T)
                    # print(batch_loss)
                    # return 0

                    ## update model
                    if self.penalty == 'l2':
                        batch_num = self.batch_num[batch_idx]
                        self.model_weights = self.model_weights - self.alpha * gradient - self.lambda_para * self.alpha * self.model_weights / batch_num
                    elif self.penalty == 'l1':
                        batch_num = self.batch_num[batch_idx]
                        self.model_weights = self.model_weights - self.alpha * gradient - self.lambda_para * self.alpha * np.sign(self.model_weights) / batch_num
                    else: 
                        # print("shape: ", self.model_weights.shape, gradient.shape)
                        self.model_weights = self.model_weights - self.alpha * gradient     # 30*1
                    # self.model_weights = self.model_weights - self.alpha * gradient     # 30*1
                # 打乱数据集的batch
                X_batch_list, y_batch_list = self.shuffle_data(X_batch_list, y_batch_list)
            
            # distributed
            else:
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

                    # print(error)
                    # print(self.gradient1.T, self.gradient2.T)
                    # print(batch_loss)
                    # return 0

                    ## update model
                    # self.model_weights = self.model_weights - self.alpha * gradient     # 30*1
                    self.weightA = self.weightA - self.alpha * self.gradient1
                    self.weightB = self.weightB - self.alpha * self.gradient2

                # 打乱数据集的batch
                X_batch_listA, X_batch_listB, y_batch_list = self.shuffle_distributed_data(X_batch_listA, 
                                    X_batch_listB, y_batch_list)
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
                    print("self.model_weights: ", self.model_weights)
                break

            self.n_iteration += 1

    

    def fit_model_distributed_input(self, X_trainA, X_trainB, Y_train, instances_count, indice_littleside):
        # indice_littleside 用于划分权重, 得到特征数值较小的那一部分的权重-或者左侧 默认X1一侧
        # mini-batch 数据集处理
        print("ratio: ", self.ratio)
        # 纵向划分数据集，位于label一侧的特征数量
        self.indice = indice_littleside # math.floor(self.ratio * ( X_trainA.shape[1]+X_trainB.shape[1] ) ) 
        # if self.ratio is None:
        #     X_batch_list, y_batch_list = self._generate_batch_data(X_train, Y_train, self.batch_size)
        if self.data_tag == None: 
            X_batch_listA, X_batch_listB, y_batch_list = self._generate_batch_data_for_distributed_parts(X_trainA, X_trainB, 
                                                                            Y_train, self.batch_size)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
            print(self.weightA.shape, self.weightB.shape)
        elif self.data_tag == 'sparse':
            print('sprase data batch generating...')
            X_batch_listA, X_batch_listB, y_batch_list = self._generate_Sparse_batch_data_for_distributed_parts(X_trainA, X_trainB, 
                                                                            Y_train, self.batch_size)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
            print('Generation done.')
        else:
            raise Exception("[fit model] No proper entry for batch data generation. Check the data_tag or the fit function.")
            
        self.n_iteration = 1
        self.loss_history = []
        test = 0
        
        while self.n_iteration <= self.max_iter:
            loss_list = []
            batch_labels = None
            # distributed
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

                # print(error)
                # print(self.gradient1.T, self.gradient2.T)
                # print(batch_loss)
                # return 0

                ## update model
                # self.model_weights = self.model_weights - self.alpha * gradient     # 30*1
                self.weightA = self.weightA - self.alpha * self.gradient1
                self.weightB = self.weightB - self.alpha * self.gradient2

            # 打乱数据集的batch
            X_batch_listA, X_batch_listB, y_batch_list = self.shuffle_distributed_data(X_batch_listA, 
                                X_batch_listB, y_batch_list)
            # print("weightA, B: ", self.weightA.shape, self.weightB.shape, self.weightA[0][0], self.weightB[0][0])
            # print("gradientA, B: ", self.gradient1.shape, self.gradient1.shape, self.gradient1[0][0], self.gradient2[0][0])
            # print("weightA, B: ", self.weightA.shape, self.weightB.shape, self.weightA[0][0], self.weightB[0][0])
            
            ## 计算 sum loss
            loss = np.sum(loss_list) / instances_count
            print("\rIteration {}, batch sum loss: {}".format(self.n_iteration, loss), end = '')
            self.loss_history.append(loss)

            ## 判断是否停止
            self.is_converged = self.check_converge_by_loss(loss)
            if self.is_converged or self.n_iteration == self.max_iter:
                # self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
                if self.ratio is not None: 
                    self.model_weights = np.hstack((self.weightA, self.weightB))
                    print("self.model_weights: ", self.model_weights)
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

        if (len(y) % batch_size > 0):
            X_tmpA, X_tmpB = np.hsplit(X[len(y) // batch_size * batch_size:, :], [self.indice])
            # X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            X_batch_listA.append(X_tmpA)
            X_batch_listB.append(X_tmpB)
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)

        return X_batch_listA, X_batch_listB, y_batch_list # listA——持有label一侧，较多样本; listB——无label一侧


    def _distributed_generate_sparse_batch_data(self, X, y, batch_size, ratio):
        '''
        sparse*
        生成两部分,纵向划分的, mini-batch数据集.
        只划分特征矩阵,不划分标签向量y
        ratio: 决定划分到含有标签的一方的数据的比例,对划分的数量下取整,例如 0.8 * 63 -> 50
        '''
        X_batch_listA = []
        X_batch_listB = []
        y_batch_list = []

        self.indice = math.floor(ratio * X.shape[1]) # 纵向划分数据集，位于label一侧的特征数量
        
        X_PartyA, X_PartyB = np.hsplit(X[:, :], [self.indice])
        X_PartyA = lil_matrix(X_PartyA)
        X_PartyB = lil_matrix(X_PartyB)

        for i in range(len(y) // batch_size):
            # X_tmpA, X_tmpB = np.hsplit(X[i * batch_size : i * batch_size + batch_size, :], [self.indice]) # 纵向划分数据集
            X_batch_listA.append(X_PartyA[i * batch_size : i * batch_size + batch_size, :].tocsr())
            X_batch_listB.append(X_PartyB[i * batch_size : i * batch_size + batch_size, :].tocsr())
            y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)

        if (len(y) % batch_size > 0):
            # X_tmpA, X_tmpB = np.hsplit(X[len(y) // batch_size * batch_size:, :], [self.indice])
            # X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            X_batch_listA.append(X_PartyA[len(y) // batch_size * batch_size:, :].tocsr())
            X_batch_listB.append(X_PartyB[len(y) // batch_size * batch_size:, :].tocsr())
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)

        return X_batch_listA, X_batch_listB, y_batch_list # listA——持有label一侧，较多样本; listB——无label一侧

    def _generate_batch_data_for_distributed_parts(self, X1, X2, y, batch_size):
        '''
        输入的数据就是两部分的
        目的是将这两部分横向ID对齐的数据 划分成一个个batch (可用于实验中的分别采样输入数据)
        ratio: 决定划分到含有标签的一方的数据的比例,对划分的数量下取整,例如 0.8 * 63 -> 50
        '''
        X_batch_listA = []
        X_batch_listB = []
        y_batch_list = []
        # self.indice = math.floor(ratio * X.shape[1]) # 纵向划分数据集，位于label一侧的特征数量
        
        for i in range(len(y) // batch_size):
            # X_tmpA = X1[i * batch_size : i * batch_size + batch_size, :]
            X_batch_listA.append(X1[i * batch_size : i * batch_size + batch_size, :])
            X_batch_listB.append(X2[i * batch_size : i * batch_size + batch_size, :])
            y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)

        if (len(y) % batch_size > 0):
            # X_tmpA, X_tmpB = np.hsplit(X[len(y) // batch_size * batch_size:, :], [self.indice])
            # X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            X_batch_listA.append(X1[len(y) // batch_size * batch_size:, :])
            X_batch_listB.append(X2[len(y) // batch_size * batch_size:, :])
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)

        return X_batch_listA, X_batch_listB, y_batch_list # listA——持有label一侧，较多样本; listB——无label一侧



    def _generate_Sparse_batch_data_for_distributed_parts(self, X1, X2, y, batch_size):
        '''
        sparse* 输入的数据就是两部分的 
        目的是将这两部分横向ID对齐的数据 划分成一个个batch (可用于实验中的分别采样输入数据)
        ratio: 决定划分到含有标签的一方的数据的比例,对划分的数量下取整,例如 0.8 * 63 -> 50
        '''
        X_batch_listA = []
        X_batch_listB = []
        y_batch_list = []
        X1 = lil_matrix(X1)
        X2 = lil_matrix(X2)
        # self.indice = math.floor(ratio * X.shape[1]) # 纵向划分数据集，位于label一侧的特征数量
        
        for i in range(len(y) // batch_size):
            # X_tmpA = X1[i * batch_size : i * batch_size + batch_size, :]
            X_batch_listA.append(X1[i * batch_size : i * batch_size + batch_size, :].tocsr())
            X_batch_listB.append(X2[i * batch_size : i * batch_size + batch_size, :].tocsr())
            y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)

        if (len(y) % batch_size > 0):
            # X_tmpA, X_tmpB = np.hsplit(X[len(y) // batch_size * batch_size:, :], [self.indice])
            # X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            X_batch_listA.append(X1[len(y) // batch_size * batch_size:, :].tocsr())
            X_batch_listB.append(X2[len(y) // batch_size * batch_size:, :].tocsr())
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)

        return X_batch_listA, X_batch_listB, y_batch_list # listA——持有label一侧，较多样本; listB——无label一侧




    def predict(self, x_test, y_test):
        # z = np.dot(x_test, self.model_weights.T)
        # z = x_test.dot(self.model_weights.T)
        # z = self._cal_z()
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T) # np.dot(features, weights.T)
        elif self.data_tag == None:
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

    
    def predict_distributed(self, x_test1, x_test2, y_test):
        # z = np.dot(x_test, self.model_weights.T)
        # z = x_test.dot(self.model_weights.T)
        # z = self._cal_z()
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T) # np.dot(features, weights.T)
        elif self.data_tag == None:
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
        print("score: ", score)
        print("len(y): ", len(y))
        rate = float(score)/float(len(y))
        print("Predict precision: ", rate)

    
    def predict_distributed_OVR(self, x_test1, x_test2):
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T)    # np.array类型（此处其实需要严谨一点，避免数据类型不清晰影响后续运算）
            if not isinstance(z, np.ndarray):
                z = z.toarray()
        elif self.data_tag == None:
            z = np.dot(x_test, self.model_weights.T)

        y = self._compute_sigmoid(z)

        return y.reshape(1, -1) # list(y.reshape((1, -1)))



    def OVRClassifier(self, X_train1, X_train2, X_test1, X_test2, Y_train, Y_test):
        """
        OVR: one vs rest 多分类
        """
        # indice_littleside = X_train1.shape[1]
        self.indice = X_train1.shape[1]
        instances_count = X_train1.shape[0]
        label_lst = list(set(Y_train))   # 多分类的所有标签值集合
        print('数据集标签值集合: ', label_lst)
        prob_lst = []                    # 存储每个二分类模型的预测概率值

        """ OVR Model Training """
        # # batch 数据生成
        # X_batch_listA, X_batch_listB, y_batch_list = self._generate_batch_data_for_distributed_parts(X_train1, X_train2, 
        #                                                                                 Y_train, self.batch_size)
        # self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
        
        for i in range(len(label_lst)):
            # 转换标签值为二分类标签值
            pos_label = label_lst[i]                                        # 选定正样本的标签
            print("Label: ", pos_label)

            # def label_reset_OVR(arr):
            #     """ 依次将标签i设置为正样本, 其他为负样本 """
            #     # global pos_label
            #     return np.where(arr == pos_label, 1, 0)
            
            # y_batch_list = list(map(label_reset_OVR, y_batch_list))
            
            Y_train_new = np.where(Y_train == pos_label, 1, 0)              # 满足条件则为正样本1，否则为负样本0
            # Y_test_new = np.where(Y_test == pos_label, 1, 0)
            # print(Y_train_new)
            self.fit_model_distributed_input(X_train1, X_train2, Y_train_new, X_train1.shape[0],
                                                                    indice_littleside)
            
            prob = self.predict_distributed_OVR(X_test1, X_test2)   # 第i个二分类模型在测试数据集上，每个样本取正标签的概率（用决策函数值作为概率值）
            prob = np.where(prob > 0, prob, 0).flatten()
            prob_lst.append(prob.tolist())
            # print(prob_lst)
        
        # 多分类模型预测
        print(np.shape(prob_lst))
        y_predict = []                      # 存储多分类的预测标签值
        prob_array = np.asarray(prob_lst).T   # (n_samples, n_classes)
        print(prob_array.shape)
        print(type(prob_array))
        print(type(prob_array[0]))
        print(type(prob_array[0][0]))

        for i in range(len(Y_test)):
            temp = list(prob_array[i])
            index = temp.index(max(temp))
            # print(index)
            y_predict.append(label_lst[index])
        # print(y_predict)
        # 模型预测准确率
        score = 0
        for i in range(len(y_predict)):
            if y_predict[i] == Y_test[i]:
                score += 1
            else:
                pass
        rate = score / len(y_predict)
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
    print(type(X_train)) # 1000 * 60
    # print(Y_train[0]) # 1000 * 1

    # 判断标签是(1;-1)还是 (1;0), 将-1的标签全部转化成0标签
    if -1 in Y_train:  
        Y_train[Y_train == -1] = 0
        Y_test[Y_test == -1] = 0
    # print(Y_train)
    # print(Y_test)

    # #a6a a7a
    # X_train = X_train.todense().A
    # X_train = np.hstack( (X_train, np.zeros(X_train.shape[0]).reshape(-1, 1)) )
    # return ss.fit_transform(X_train), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array

    # #splice
    # return ss.fit_transform(X_train.todense().A), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array
    # # return X_train.todense().A, Y_train, X_test.todense().A, Y_test # matrix转array
    

    # sparse matrix (splice)
    # return ss.fit_transform(X_train), Y_train, ss.fit_transform(X_test), Y_test # matrix转array
    return X_train, Y_train, X_test, Y_test # matrix转array


    # X = np.loadtxt('/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/X_train_sketch.txt', 
    #                 delimiter=',') #, dtype = float)


def read_encoded_data():
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
    # X_train = train_data[0]
    Y_train = train_data[1].astype(int)
    # X_test = test_data[0]
    Y_test = test_data[1].astype(int)
    # print(type(X_train)) # 1000 * 60
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
    # return ss.fit_transform(X_train.todense().A), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array
    # # return X_train.todense().A, Y_train, X_test.todense().A, Y_test # matrix转array
    print("loading dataset...")
    X_train = np.loadtxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/encoded/X_encoded_train37.txt", delimiter=',', dtype = int)
    X_test = np.loadtxt("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/splice/distrubuted/encoded/X_encoded_test37.txt", delimiter=',', dtype = int)
    # X = normalize(X,'l2')
    # X_train = ss.fit_transform(X_train)
    print(X_train.shape)         #查看特征形状
    print(type(X_train), type(X_test))
    print(X_test.shape)         #查看测试特征形状



    # print("Constructing sparse matrix...") # 使用COO格式高效创建稀疏矩阵, 以线性时间复杂度转化为CSR格式用于高效的矩阵乘法或转置运算.
    # X_train = lil_matrix(X_train)
    # # X_train.tocsr()
    # X_test = lil_matrix(X_test)
    # # X_test.tocsr()
    # print(type(X_train), type(X_test))
    
    return X_train, Y_train, X_test, Y_test # matrix转array



def read_distributed_encoded_data():
    from sklearn.datasets import load_svmlight_file
    import os
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
    mm = MinMaxScaler()
    ss = StandardScaler()

    dataset_file_name = 'splice'  
    train_file_name = 'splice_train.txt' 
    test_file_name = 'splice_test'
    main_path = PATH_DATA

    # dataset_file_name = 'a6a'
    # train_file_name = 'a6a.txt'
    # test_file_name = 'a6a.t'
    # main_path = '/Users/zbz/data/'
    train_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, train_file_name))
    test_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, test_file_name))
    # X_train = train_data[0]
    Y_train = train_data[1].astype(int)
    # X_test = test_data[0]
    Y_test = test_data[1].astype(int)
    # print(type(X_train)) # 1000 * 60
    # print(Y_train[0]) # 1000 * 1

    # 判断标签是(1;-1)还是 (1;0), 将-1的标签全部转化成0标签
    if -1 in Y_train:  
        Y_train[Y_train == -1] = 0
        Y_test[Y_test == -1] = 0
    # print(Y_train)
    # print(Y_test)

    # #a6a a7a
    # X_train = X_train.todense().A
    # X_train = np.hstack( (X_train, np.zeros(X_train.shape[0]).reshape(-1, 1)) )
    # return ss.fit_transform(X_train), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array

    # #splice
    # return ss.fit_transform(X_train.todense().A), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array
    # # return X_train.todense().A, Y_train, X_test.todense().A, Y_test # matrix转array
    print("loading dataset...")

    dataset_file_name = 'splice/distrubuted/encoded/'  
    train_file_name1 = 'X1_encoded_train37.txt'
    train_file_name2 = 'X2_encoded_train37.txt'
    test_file_name1 = 'X1_encoded_test37.txt'
    test_file_name2 = 'X2_encoded_test37.txt'
    main_path = PATH_DATA
    X_train1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',') #, dtype = float)
    X_train2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',') #, dtype = float)
    X_test1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',') #, dtype = float)
    X_test2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',') #, dtype = float)
    # X = normalize(X,'l2')
    # X_train = ss.fit_transform(X_train)
    print(X_train1.shape)         #查看特征形状
    print(type(X_train1), type(X_test1))
    print(X_test1.shape)         #查看测试特征形状

    # print("Constructing sparse matrix...") # 使用COO格式高效创建稀疏矩阵, 以线性时间复杂度转化为CSR格式用于高效的矩阵乘法或转置运算.
    # X_train = lil_matrix(X_train)
    # # X_train.tocsr()
    # X_test = lil_matrix(X_test)
    # # X_test.tocsr()
    # print(type(X_train), type(X_test))
    
    return X_train1, X_train2, Y_train, X_test1, X_test2, Y_test # matrix转array


def read_distributed_squeeze_data():
    ## countsketch
    from sklearn.datasets import load_svmlight_file
    import os
    global flag
    flag = "sketch"

    dataset_file_name = 'DailySports'  
    train_file_name = 'DailySports_train.txt' 
    test_file_name = 'DailySports_test.txt'
    main_path = PATH_DATA
    # main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'

    # dataset_file_name = 'a6a'
    # train_file_name = 'a6a.txt'
    # test_file_name = 'a6a.t'
    # main_path = '/Users/zbz/data/'
    train_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, train_file_name))
    test_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, test_file_name))
    # X_train = train_data[0]
    Y_train = train_data[1].astype(int)
    # X_test = test_data[0]
    Y_test = test_data[1].astype(int)
    # print(type(X_train)) # 1000 * 60
    # print(Y_train[0]) # 1000 * 1

    ##### 判断标签是(1;-1)还是 (1;0), 将-1的标签全部转化成0标签
    # if -1 in Y_train:  
    #     Y_train[Y_train == -1] = 0
    #     Y_test[Y_test == -1] = 0
    
    # 针对SprotsNews, 多分类修改成二分类
    print("processing dataset...")
    # Y_train[Y_train != 1] = 0
    # Y_test[Y_test != 1] = 0

    # #a6a a7a
    # X_train = X_train.todense().A
    # X_train = np.hstack( (X_train, np.zeros(X_train.shape[0]).reshape(-1, 1)) )
    # return ss.fit_transform(X_train), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array

    # #splice
    # return ss.fit_transform(X_train.todense().A), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array
    # # return X_train.todense().A, Y_train, X_test.todense().A, Y_test # matrix转array
    print("loading dataset...")

    dataset_file_name = 'DailySports/portion37_pminhash/sketch1024/countsketch/'
    train_file_name1 = 'X1_squeeze_train37_Countsketch.txt'
    train_file_name2 = 'X2_squeeze_train37_Countsketch.txt'
    test_file_name1 = 'X1_squeeze_test37_Countsketch.txt'
    test_file_name2 = 'X2_squeeze_test37_Countsketch.txt'
    # main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'
    main_path = PATH_DATA
    X_train1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',') #, dtype = float)
    X_train2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',') #, dtype = float)
    X_test1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',') #, dtype = float)
    X_test2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',') #, dtype = float)
    # X = normalize(X,'l2')
    # X_train = ss.fit_transform(X_train)
    print(X_train1.shape)         #查看特征形状
    print(type(X_train1), type(X_test1))
    print(X_test1.shape)         #查看测试特征形状

    # print("Constructing sparse matrix...") # 使用COO格式高效创建稀疏矩阵, 以线性时间复杂度转化为CSR格式用于高效的矩阵乘法或转置运算.
    # X_train = lil_matrix(X_train)
    # # X_train.tocsr()
    # X_test = lil_matrix(X_test)
    # # X_test.tocsr()
    # print(type(X_train), type(X_test))
    
    return X_train1, X_train2, Y_train, X_test1, X_test2, Y_test # matrix转array


# def read_distributed_squeeze_data():
#     ## countsketch
#     from sklearn.datasets import load_svmlight_file
#     import os

#     dataset_file_name = 'kits'  
#     train_file_name = 'kits_train.txt' 
#     test_file_name = 'kits_test.txt'
#     # dataset_file_name = 'DailySports'  
#     # train_file_name = 'DailySports_train.txt' 
#     # test_file_name = 'DailySports_test.txt'
#     main_path = PATH_DATA
#     # main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'

#     # dataset_file_name = 'a6a'
#     # train_file_name = 'a6a.txt'
#     # test_file_name = 'a6a.t'
#     # main_path = '/Users/zbz/data/'
#     train_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, train_file_name))
#     test_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, test_file_name))
#     # X_train = train_data[0]
#     Y_train = train_data[1].astype(int)
#     # X_test = test_data[0]
#     Y_test = test_data[1].astype(int)
#     # print(type(X_train)) # 1000 * 60
#     # print(Y_train[0]) # 1000 * 1

#     ##### 判断标签是(1;-1)还是 (1;0), 将-1的标签全部转化成0标签
#     # if -1 in Y_train:  
#     #     Y_train[Y_train == -1] = 0
#     #     Y_test[Y_test == -1] = 0
    
#     # 针对SprotsNews, 多分类修改成二分类
#     print("processing dataset...")
#     Y_train[Y_train != 1] = 0
#     Y_test[Y_test != 1] = 0
#     # print(Y_train)
#     # print(Y_test)

#     # #a6a a7a
#     # X_train = X_train.todense().A
#     # X_train = np.hstack( (X_train, np.zeros(X_train.shape[0]).reshape(-1, 1)) )
#     # return ss.fit_transform(X_train), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array

#     # #splice
#     # return ss.fit_transform(X_train.todense().A), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array
#     # # return X_train.todense().A, Y_train, X_test.todense().A, Y_test # matrix转array
#     print("loading dataset...")

#     dataset_file_name = 'kits/portion37_pminhash/sketch1024/countsketch/'
#     train_file_name1 = 'X1_squeeze_train37_Countsketch.txt'
#     train_file_name2 = 'X2_squeeze_train37_Countsketch.txt'
#     test_file_name1 = 'X1_squeeze_test37_Countsketch.txt'
#     test_file_name2 = 'X2_squeeze_test37_Countsketch.txt'
#     # main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'
#     main_path = PATH_DATA
#     X_train1 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name1), delimiter=',') #, dtype = float)
#     X_train2 = np.loadtxt(os.path.join(main_path, dataset_file_name, train_file_name2), delimiter=',') #, dtype = float)
#     X_test1 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name1), delimiter=',') #, dtype = float)
#     X_test2 = np.loadtxt(os.path.join(main_path, dataset_file_name, test_file_name2), delimiter=',') #, dtype = float)
#     # X = normalize(X,'l2')
#     # X_train = ss.fit_transform(X_train)
#     # print(X_train1.shape)         #查看特征形状
#     # print(type(X_train1), type(X_test1))
#     # print(X_test1.shape)         #查看测试特征形状
#     print("X_train1 type: ", type(X_train1)) # 1000 * 60
#     print("X_train1 shape: ", X_train1.shape)
#     print("X_train2 type: ", type(X_train2)) # 1000 * 60
#     print("X_train2 shape: ", X_train2.shape)
#     print("X_test1 type: ", type(X_test1)) # 1000 * 60
#     print("X_test1 shape: ", X_test1.shape)
#     print("X_test2 type: ", type(X_test2)) # 1000 * 60
#     print("X_test2 shape: ", X_test2.shape)

#     # print("Constructing sparse matrix...") # 使用COO格式高效创建稀疏矩阵, 以线性时间复杂度转化为CSR格式用于高效的矩阵乘法或转置运算.
#     # X_train = lil_matrix(X_train)
#     # # X_train.tocsr()
#     # X_test = lil_matrix(X_test)
#     # # X_test.tocsr()
#     # print(type(X_train), type(X_test))
    
#     return X_train1, X_train2, Y_train, X_test1, X_test2, Y_test # matrix转array

if __name__ == "__main__":
    ########## 读取数据 ##########
    # 基础测试
    # X_data, y_data, X_test, y_test = read_data()
    # X_data, y_data, X_test, y_test = read_sampling_data()
    # 理想的集中数据集
    # X_data, y_data, X_test, y_test = read_encoded_data()
    # print(X_data.shape, X_data.shape[0], X_data.shape[1], y_data.shape, X_test.shape, y_test.shape)

    # 纵向划分的数据集
    # X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_encoded_data()
    X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_squeeze_data()
    # print(X_train1.shape, X_train2.shape, X_train1.shape[1], X_train2.shape[1], Y_train.shape, X_test1.shape, Y_test.shape)

    ########## 权重初始化 ##########
    np.random.seed(100)
    # # 正态
    # weight_vector = np.random.normal(0.0, 0.0001, X_data.shape[1]).reshape(1, -1)
    # # 全1
    # weight_vector = np.ones(X_data.shape[1]).reshape(1,-1)
    # # 随机
    # weight_vector = np.random.random(X_data.shape[1]).reshape(1, -1)
    # weight = (0.00001-0.0) * weight_vector + 0.0
    # # 全0
    # weight_vector = np.zeros(X_data.shape[1]).reshape(1, -1)

    # 纵向划分分布式
    weight_vector = np.zeros(X_train1.shape[1]+X_train2.shape[1]).reshape(1, -1)
    # print(weight_vector)


    ########## 模型实例化 ##########
    # 伪纵向划分——检验算法正确性
    # LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
    #                 max_iter = 1000, alpha = 0.0001, eps = 1e-6, ratio = 0.7, data_tag = 'sparse')  # 0.956140350877193
                    # splice 分布式: Predict precision:  0.8496551724137931
                    # splice 分布式 0.9062068965517242
    # 理想集中
    # LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
    #                 max_iter = 2000, alpha = 0.0001, eps = 1e-6, penalty = None, lambda_para = 1, data_tag='sparse')  
                    # breast_cancer: 0.9649122807017544
                    # splice: 0.8482758620689655
                    # splice 集中 0.9062068965517242
    # 纵向划分分布式(这里的ratio控制的是weight的划分比例, 需要根据输入的数据的划分比例手动确定)——ratio没有用, 容易和数据集的两部分数量不对应, 后面还要修改
    LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 32, 
                    max_iter = 500, alpha = 0.001, eps = 1e-5, ratio = 0.7, penalty = None, lambda_para = 1, data_tag = 'sparse')
                    # splice 分布式 0.9062068965517242

    # 两部分数据集
    # sparse: 12.54981803894043 s,       Predict precision:  0.9062068965517242    Iteration 645
    # non-sparse: 30.390305995941162 s,  Predict precision:  0.9062068965517242    Iteration 645

    # 集中：
    # sparse:  14.546779870986938 s   Predict precision:  0.9062068965517242

    ########## 训练 ##########
    import time
 
    time_start = time.time()
    
    # 理想集中和伪分布式
    # LogisticRegressionModel.fit_model(X_data, y_data, X_data.shape[0])

    # 纵向划分分布式
    indice_littleside = X_train1.shape[1] # 纵向A的特征数量, 
    # LogisticRegressionModel.fit_model_distributed_input(X_train1, X_train2, Y_train, X_train1.shape[0], indice_littleside)
    LogisticRegressionModel.OVRClassifier(X_train1, X_train2, X_test1, X_test2, Y_train, Y_test)
    time_end = time.time()
    print('time cost: ',time_end-time_start,'s')

    # plt.plot(LogisticRegressionModel.loss_history)
    # plt.show()

    ########## 测试 ##########
    # 理想集中和伪分布式
    # LogisticRegressionModel.predict(X_test, y_test)
    # 纵向划分分布式
    # LogisticRegressionModel.predict_distributed(X_test1, X_test2, Y_test)
