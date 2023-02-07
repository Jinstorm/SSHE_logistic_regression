import numpy as np
# import pickle

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
        self.batch_num = [] # 存储每个batch都大小
        self.n_iteration = 0
        self.max_iter = max_iter
        self.alpha = alpha
        self.pre_loss = 0
        self.eps = eps


    def _cal_z(self, weights, features):
        # if not self.reveal_every_iter:
        #     # LOGGER.info(f"[forward]: Calculate z in share...")
        #     w_self, w_remote = weights
        #     z = self._cal_z_in_share(w_self, w_remote, features, suffix, cipher)
        self.wx_self = np.dot(features, weights.T)
        print("self.wx_self shape:", self.wx_self.shape)
        # print("self.wx_self: ", self.wx_self)
        # return self.wx_self

    def _compute_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        # return z * 0.25 + 0.5 

    def _compute_loss_cross_entropy(self, weights, label, batch_idx):
        """
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
        """
        # self.wx_self = np.dot(weights, x)
        half_wx = -0.5 * self.wx_self
        # print("wx_self: ", self.wx_self.shape, "label:", label.reshape(-1, 1).shape)
        ywx = self.wx_self * label #np.dot(label, self.wx_self)
        wx_square = self.wx_self * self.wx_self * -0.125 # wx_square = np.dot(self.wx_self.T, self.wx_self) * -0.125 # 这里后续要修改，两方平方，有交叉项
        
        # print("wx_square: ", wx_square.shape)
        # print("ywx: ", ywx.shape)

        batch_num = self.batch_num[batch_idx]
        print("batch_num:",batch_num)

        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # tmp = (half_wx + ywx + wx_square) * (-1 / batch_num)
        # print("sum:llllll: ", tmp, np.sum(tmp))
        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )

        # if loss_norm == 'l2_penalty':
        # print("loss: ", loss.shape)
        # print("loss:",loss)
        return loss

    def _compute_loss(self, y, label, batch_idx):
        batch_num = self.batch_num[batch_idx]
        loss = -1 * label * np.log(y) - (1 - label) * np.log(1-y)
        return np.sum(loss)




    def forward(self, weights, features, labels): #, batch_weight):
        self._cal_z(weights, features)
        # sigmoid
        sigmoid_z = self._compute_sigmoid(self.wx_self)
        return sigmoid_z

    def backward(self, error, features, length):
        print("Backward: ********************************")
        print("error, feature shape: ", error.shape, features.shape)
        g = np.dot(error.T, features) / length
        return g

    def check_converge_by_loss(self, loss):
        converge_flag = False
        if self.pre_loss is None:
            pass
        elif abs(self.pre_loss - loss) < self.eps:
            converge_flag = True
        self.pre_loss = loss
        return converge_flag

    def fit_model(self, X_train, Y_train, instances_count):

        X_batch_list, y_batch_list = self._generate_batch_data(X_train, Y_train, self.batch_size)

        self.n_iteration = 0
        self.loss_history = []
        test = 0
        
        while self.n_iteration < self.max_iter:
            loss_list = []
            batch_labels = None

            for batch_idx, batch_data in enumerate(X_batch_list):
                test += 1
                if test >= 4: break
                print("ID:    ", batch_idx)
                batch_labels = y_batch_list[batch_idx].reshape(-1, 1)
                batch_labels[batch_labels == 0] = 0.01
                batch_labels[batch_labels == 1] = 0.99
                
                ## forward and backward
                # print("weight: {} batch_data: {}batch_labels: {}".format(self.model_weights, batch_data, batch_labels))
                print("model weight shape: ", self.model_weights.shape)
                y = self.forward(self.model_weights, batch_data, batch_labels)
                print("y shape: ", y.shape, "batch label shape: ", batch_labels.shape)
                print("y_predict: ", y)
                print("batch_labels: ", batch_labels)

                error = y - batch_labels # 注意这里谁减谁，和后面更新梯度是加还是减有关
                print("error shape: ", error.shape)
                # print("y, labels: ", y, batch_labels)
                gradient = self.backward(error = error, features = batch_data, batch_idx = batch_idx)
                print("gradient shape: ", gradient.shape)
                print("gradient: ", gradient)

                ## compute loss
                batch_loss = self._compute_loss_cross_entropy(weights = self.model_weights, label = batch_labels, batch_idx = batch_idx)
                # batch_loss = self._compute_loss(y, batch_labels, batch_idx)
                # if batch_loss is not None:
                    # batch_loss = batch_loss * self.batch_num[batch_idx] # 因为方便后面计算sum loss直接除以数据集的样本总量，误差函数是所有样本损失函数的算数平均值。
                    # print("batch_loss: ", batch_loss)
                loss_list.append(batch_loss)

                ## update model
                print("weight.shape: ", self.model_weights.shape)
                print("model: ", self.model_weights)
                
                self.model_weights = self.model_weights - self.alpha * gradient     # 30*1
                # print("Model: ", self.model_weights, "\ngradient: ", gradient)
                print("——————————————————————————————###")
            
            print("loss list: ", loss_list[0], loss_list[1], loss_list[2])

            ## 计算 sum loss
            loss = np.sum(loss_list) / instances_count
            print("\rIteration {}, batch sum loss: {}".format(self.n_iteration, loss))
            self.loss_history.append(loss)

            ## 判断是否停止
            self.is_converged = self.check_converge_by_loss(loss)
            # if self.stop_training:
            #     break
            if self.is_converged:
                break

            self.n_iteration += 1


    # def batch_data_generator(features, labels, batch_size):
    #     batch_labels_list = []
    #     batch_weight_list = []

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

        print("batch_num: ", self.batch_num)
        print(X_batch_list[0].shape, y_batch_list)
        return X_batch_list, y_batch_list

def read_data():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    cancers = load_breast_cancer()
    X = cancers.data       #获取特征值
    Y = cancers.target     #获取标签
    print(X.shape)         #查看特征形状
    print(Y.shape)         #查看标签形状
    # print(X)
    # print(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return x_train, y_train
    # f = open(path, "r")

    # X = list()
    # y = list()

    # encode_char = ""
    # encoder = lambda y : 0 if y is encode_char else 1

    # for row in f:
    #     split_row = [x.strip() for x in row.split(',')]
    #     if encode_char is "":
    #         encode_char = split_row[-1]
    #     y.append(encoder(split_row[-1]))
    #     X.append([np.array(split_row[:-1]).astype(np.float)])

    # permutations = np.random.permutation(len(X))

    # X, y = np.asarray(X).squeeze(), np.asarray(y)

    # X = X[permutations, :]
    # y = y[permutations]

    # #To add beta 0
    # temp = np.ones((X.shape[0], X.shape[1] + 1))
    # temp[:, 1:] = X
    # X = temp

    # len_test = len(X) // 5 
    # len_train = len(X) - len_test
    # X_test, y_test, X_train, y_train = X[:len_test, :], y[:len_test], X[len_test:, :], y[len_test:]
    # print(X_train)
    # return X_train, y_train

if __name__ == "__main__":
    X_data, y_data = read_data()
    print(X_data.shape, X_data.shape[0], X_data.shape[1], y_data.shape)
    # weight_vector = np.random.random(X_data.shape[1]).reshape(1, -1)
    # print(weight_vector)

    weight_vector = np.random.normal(0.0, 0.0001, X_data.shape[1]).reshape(1, -1)
    print(weight_vector.shape)
    # 模型实例化
    LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 10, max_iter = 1, alpha = 0.001, eps = 1e-2)
    # 训练
    LogisticRegressionModel.fit_model(X_data, y_data, X_data.shape[0])