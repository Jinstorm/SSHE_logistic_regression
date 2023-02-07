'''
# 小批量梯度下降逻辑回归实现 mini-batch logistic regression 2022.11.27


# 删掉了多余的可用输出和可用操作, 标准化输入之后在sklearn的breast数据集上准确率90%+
# 激活函数: 使用minmax近似sigmoid函数, sig = z * 0.25 + 0.5
# 损失函数: 交叉熵损失函数, 使用Taylor近似
# 绘图: 主函数中有绘图函数, 暂时注释掉了, 取消两行的注视即可

'''
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    logistic回归
    """
    def __init__(self, weight_vector, batch_size, max_iter, alpha, eps):
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
        self.eps = eps


    def _cal_z(self, weights, features):
        self.wx_self = np.dot(features, weights.T)
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
        half_wx = -0.5 * self.wx_self
        ywx = self.wx_self * label

        wx_square = self.wx_self * self.wx_self * -0.125 # wx_square = np.dot(self.wx_self.T, self.wx_self) * -0.125 # 这里后续要修改，两方平方，有交叉项
        batch_num = self.batch_num[batch_idx]

        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )
        return loss

    def _compute_loss(self, y, label, batch_idx):
        batch_num = self.batch_num[batch_idx]
        loss = -1 * label * np.log(y) - (1 - label) * np.log(1 - y)
        return np.sum(loss)


    def forward(self, weights, features, labels): #, batch_weight):
        self._cal_z(weights, features)
        # sigmoid
        sigmoid_z = self._compute_sigmoid(self.wx_self)
        return sigmoid_z

    def backward(self, error, features, batch_idx):
        batch_num = self.batch_num[batch_idx]
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

    def fit_model(self, X_train, Y_train, instances_count):
        # mini-batch 数据集处理
        X_batch_list, y_batch_list = self._generate_batch_data(X_train, Y_train, self.batch_size)

        self.n_iteration = 0
        self.loss_history = []
        test = 0
        
        while self.n_iteration < self.max_iter:
            loss_list = []
            batch_labels = None

            for batch_idx, batch_data in enumerate(X_batch_list):
                batch_labels = y_batch_list[batch_idx].reshape(-1, 1)
                
                ## forward and backward
                y = self.forward(self.model_weights, batch_data, batch_labels)
                error = y - batch_labels # 注意这里谁减谁，和后面更新梯度是加还是减有关
                gradient = self.backward(error = error, features = batch_data, batch_idx = batch_idx)

                ## compute loss
                batch_loss = self._compute_loss_cross_entropy(weights = self.model_weights, label = batch_labels, batch_idx = batch_idx)
                # batch_loss = self._compute_loss(y, batch_labels, batch_idx)
                loss_list.append(batch_loss)

                ## update model
                self.model_weights = self.model_weights - self.alpha * gradient     # 30*1

            ## 计算 sum loss
            loss = np.sum(loss_list) / instances_count
            print("\rIteration {}, batch sum loss: {}".format(self.n_iteration, loss))
            self.loss_history.append(loss)

            ## 判断是否停止
            self.is_converged = self.check_converge_by_loss(loss)
            if self.is_converged:
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
        ratio = score/len(y)
        print(ratio)

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


if __name__ == "__main__":
    X_data, y_data, X_test, y_test = read_data()
    print(X_data.shape, X_data.shape[0], X_data.shape[1], y_data.shape)
    np.random.seed(100)
    weight_vector = np.random.normal(0.0, 0.0001, X_data.shape[1]).reshape(1, -1)

    # print(weight_vector)
    # 模型实例化
    LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
                max_iter = 1000, alpha = 0.0001, eps = 1e-6)
    # 训练
    LogisticRegressionModel.fit_model(X_data, y_data, X_data.shape[0])

    # 历史损失值绘图，取消下面两行注视即可
    # plt.plot(LogisticRegressionModel.loss_history)
    # plt.show()
    LogisticRegressionModel.predict(X_test, y_test)