import numpy as np
 
 
class LogisticRegression:
    """
    Logistic Regression:逻辑回归
    Author：CommissarMa
    """
 
    def __init__(self, m, n, X, y, alpha, iterThreshold):
        """
        构造函数：初始化成员变量
        :param m: 记录数量（int）
        :param n: 特征数量（int）
        :param X: 记录矩阵（n*m）（float）
        :param y: 类别向量（1*m）（取值范围：0或1）
        :param alpha: 更新速率（float）
        :param iterThreshold: 梯度下降迭代停止条件（float）
        :var w: 参数向量（n*1）（float）
        :var b: 参数（float）
        """
        self.m = m
        self.n = n
        self.X = X
        self.y = y
        self.alpha = alpha
        self.iterThreshold = iterThreshold
        self.w = np.zeros((n, 1))
        self.b = 0
        return
 
    def train(self):
        """
        训练：使用数据进行训练，使用梯度下降进行迭代使得损失值不断下降直到小于设定的迭代停止条件
        :return:训练完成后得到最优的w和b
        """
        JLast = -1  # 用来存放上一次迭代的损失值。用-1是因为损失值>=0
        count = 0  # 迭代次数
        while True:
            count += 1
            J = 0  # 损失值
            dw = np.zeros((self.n, 1))  # a对w的导数（n*1）
            db = 0  # a对b的导数
            Z = np.dot(self.w.T, self.X) + self.b  # Z=wT*X+b
            a = 1 / (1 + np.exp(-Z))  # Sigmoid函数
            J += -(np.dot(self.y, np.log(a).T) + np.dot(1 - self.y, np.log(1 - a).T))  # 损失函数的计算
            dz = a - self.y  # a对z的导数（m*1）
            dw += np.dot(self.X, dz.T)
            db += np.sum(dz, axis=1)
            J /= self.m  # 平均损失
            dw /= self.m
            db /= self.m
            self.w -= self.alpha * dw
            self.b -= self.alpha * db
            print("第" + str(count) + "次梯度下降的损失值J：" + str(J))
            if np.abs(J - JLast) < self.iterThreshold and JLast > 0:
                break
            JLast = J
        return self.w, self.b
 
    def predict(self, x):
        """
        预测:对新的记录进行预测，给出预测的类别
        :param x:需要进行预测的一条记录（n*1）
        :return:如果预测出的概率大于0.5就返回类别1，小于等于0.5就返回类别0
        """
        result = np.dot(self.w.T, x) + self.b
        result = 1 / (1 + np.exp(-result))
        if result > 0.5:
            return 1
        else:
            return 0
 
 
if __name__ == '__main__':
    m = 3  # 样本个数
    n = 2  # 样本特征数
    X = np.random.rand(m * n)  # 随机生成样本
    X = X.reshape(n, m)  # 转成n行m列
    y = np.random.randint(0, 2, (1, m))  # 1行m列的y向量（y属于0或1）
    alpha = 0.1  # 设置更新速率
    iterThreshold = 0.00001  # 设置迭代停止条件
    lr = LogisticRegression(m, n, X, y, alpha, iterThreshold)
    lr.train()
    print(lr.predict(np.array([np.random.rand(1), np.random.rand(1)])))
    print(help(LogisticRegression))