'''
TODO:
time accounting.
'''
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from multiprocessing import Pool

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_pardir = os.path.join(dir_path, os.pardir)
abs_parpardir = os.path.join(abs_pardir, os.pardir)
sys.path.append(abs_parpardir)
# print(abs_parpardir)
from paillierm.encrypt import PaillierEncrypt
from paillierm.fixedpoint import FixedPointEndec
from paillierm.utils import urand_tensor
# from federatedml.secureprotol import PaillierEncrypt
# from federatedml.secureprotol.fixedpoint import FixedPointEndec
# from federatedml.secureprotol.spdz.utils import urand_tensor

PATH_DATA = '../../data/' # '../../data/'
# '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'
# '/data/projects/fate/SSHE/data'
# main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'

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

        # WAN(Wide area network) Bandwidth, unit: 使用单位: Mbps (1 MB/s = 8 Mbps); 带宽测试: 40Mbps (5MB/s)
        self.WAN_bandwidth = 40 # Mbps
        self.train_time_account = 0
        self.mem_occupancy = 4 # B 字节 
        # 计算时: 元素个数 * 4 B / 1024 / 1024 MB  / (40/8) s = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)

        # 加密部分的初始化
        self.cipher = PaillierEncrypt() # Paillier初始化
        self.cipher.generate_key()  # Paillier生成公私钥
        self.fixedpoint_encoder = FixedPointEndec(n = 1e10) # 加密前的定点化编码器初始化

        # 进程池
        # self.pool = Pool()



    def _cal_z(self, weights, features, party = None, encrypt = None):
        if encrypt is not None:
            if party == "A":  
                self.za1 = np.dot(features, weights.T)
                # print("za1: ", self.za1)
            elif party == "B": 
                self.zb2 = np.dot(features, weights.T)
                # print("zb2: ", self.zb2)
            else: raise NotImplementedError

        elif party == "A":
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
        self._cal_z(weights, features, party = None, encrypt = None)
        # sigmoid
        sigmoid_z = self._compute_sigmoid(self.wx_self)
        return sigmoid_z

    def distributed_forward(self, weights, features, party = None): #, batch_weight):
        # print("party: ", party)
        self._cal_z(weights, features, party, encrypt = None)
        # sigmoid
        if party == "A":
            sigmoid_z = self._compute_sigmoid(self.wx_self_A)
        elif party == "B":
            # sigmoid_z = self._compute_sigmoid(self.wx_self_B)
            # 注意这里考虑到分布式的计算问题, 少加了一个0.5, 这样后面y1+y2-label计算loss的时候才是正确的error值
            sigmoid_z = self._compute_sigmoid_dual_distributed(self.wx_self_B)
            
        return sigmoid_z

    def backward(self, error, features, batch_idx):
        batch_num = self.batch_num[batch_idx]
        if self.data_tag == 'sparse':
            gradient = features.T.dot(error).T / batch_num # 稀疏矩阵
        elif self.data_tag == None:
            gradient = np.dot(error.T, features) / batch_num # 非稀疏矩阵
        # print("gradient shape: ", gradient.shape)
        return gradient

    def distributed_backward(self, error, features, batch_num):
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
    
    def shuffle_distributed_data(self, XdatalistA, XdatalistB, Ydatalist):
        zip_list = list( zip(XdatalistA, XdatalistB, Ydatalist) )              # 将a,b整体作为一个zip,每个元素一一对应后打乱
        np.random.shuffle(zip_list)               # 打乱c
        XdatalistA[:], XdatalistB[:], Ydatalist[:] = zip(*zip_list)
        return XdatalistA, XdatalistB, Ydatalist


    def time_counting(self, tensor):
        # 计算tensor在WAN下传输的时间
        if tensor.ndim == 2:
            object_num = tensor.shape[0] * tensor.shape[1]
        else:
            object_num = tensor.shape[0]
        commTime = object_num * self.mem_occupancy / (1024*1024) / (self.WAN_bandwidth/8)
        self.train_time_account += commTime

    def secret_share_vector_plaintext(self, share_target):
        '''
        Desc: 秘密分享(输入的share_target是明文)
        '''
         # 生成本地share向量
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
        share = share_target - tmp
        self.time_counting(share)
        return tmp, share # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share


    def secret_share_vector(self, share_target):
        '''
        Desc: 秘密分享(输入的share_target是个加密的)
        '''
        # 生成本地share向量
        # print("share_target shape: ", share_target.shape)
        
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
        # print("pre shape: ", _pre.shape)
        share = share_target - tmp
        self.time_counting(share)
        return tmp, self.cipher.recursive_decrypt(share) # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share

    def secure_Matrix_Multiplication(self, matrix, vector, stage = None):
        '''
        输入:   数据矩阵和向量
        返回值: share的两个分片
        向量加密后做矩阵乘法, 然后 secret share 乘积结果的矩阵分成2个sharings
        '''
        # import time
 
        # time_start = time.time()
    
        if stage == "forward":
            encrypt_vec = self.cipher.recursive_encrypt(vector)
            # encrypt_vec = np.asarray(self.pool.map(self.cipher.recursive_encrypt, vector))
            # self.pool.close()
            # self.pool.join()
            # print("forward vector shape: ", vector.shape)
            assert(matrix.shape[1] == encrypt_vec.shape[1])
            mul_result = np.dot(matrix, encrypt_vec.T)
        elif stage == "backward":
            encrypt_vec = self.cipher.recursive_encrypt(vector)
            # encrypt_vec = np.asarray(self.pool.map(self.cipher.recursive_encrypt, vector))
            # self.pool.close()
            # print("backward vector shape: ", vector.shape)
            assert(encrypt_vec.shape[0] == matrix.shape[0])
            mul_result = np.dot(encrypt_vec.T, matrix)

        else: raise NotImplementedError
        return self.secret_share_vector(mul_result)

    def secure_distributed_cal_z(self, X, w1, w2, party = None):
        """
        Do the X·w and split into two sharings.

        Parameters
        ----------
        X: ndarray - numpy
           data to use for multiplication
        w1: ndarray ``1 * m1``
           piece 1 of the model weight
        w2: ndarray ``1 * m2``
           piece 2 of the model weight

        Returns
        -------
        Two sharings of the result (X·w)
        """
        if party == "A":
            self._cal_z(X, w1, party = party, encrypt = "paillier")
            assert(X.shape[1] == w2.shape[1])
            self.za2_1, self.za2_2 = self.secure_Matrix_Multiplication(X, w2, stage = "forward")

        elif party == "B":
            self._cal_z(X, w2, party = party, encrypt = "paillier")
            self.zb1_1, self.zb1_2 = self.secure_Matrix_Multiplication(X, w1, stage = "forward")
        else: raise NotImplementedError


    def secure_distributed_compute_loss_cross_entropy(self, label, batch_num):
        """
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
        """
        # self.encrypted_wx = self.wx_self_A + self.wx_self_B
        half_wx = -0.5 * self.encrypt_wx
        assert(self.encrypt_wx.shape[0] == label.shape[0])
        ywx = self.encrypt_wx * label

        # print()
        wx_square = (self.za * self.za + 2 * self.za * self.zb + self.zb * self.zb) * -0.125
        # wx_square = (2*self.wx_self_A * self.wx_self_B + self.wx_self_A * self.wx_self_A + self.wx_self_B * self.wx_self_B) * -0.125 
        # wx_square = np.dot(self.wx_self.T, self.wx_self) * -0.125 # 这里后续要修改，两方平方，有交叉项
        # wx_square2 = self.encrypted_wx * self.encrypted_wx * -0.125
        # assert all(wx_square == wx_square2)  # 数组比较的返回值为: 类似[True False False]

        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )
        return loss


    def fit_model(self, X_trainA, X_trainB, Y_train, instances_count, indice_littleside):
        # indice_littleside 用于划分权重, 得到特征数值较小的那一部分的权重-或者左侧 默认X1一侧
        # mini-batch 数据集处理
        # print("ratio: ", self.ratio)
        self.indice = indice_littleside # math.floor(self.ratio * ( X_trainA.shape[1]+X_trainB.shape[1] ) ) # 纵向划分数据集，位于label一侧的特征数量
        # if self.ratio is None:
        #     X_batch_list, y_batch_list = self._generate_batch_data(X_train, Y_train, self.batch_size)
        if self.data_tag == None: 
            X_batch_listA, X_batch_listB, y_batch_list = self._generate_batch_data_for_distributed_parts(X_trainA, X_trainB, Y_train, self.batch_size)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
            # print(self.weightA.shape, self.weightB.shape)
        elif self.data_tag == 'sparse':
            print('sprase data batch generating...')
            X_batch_listA, X_batch_listB, y_batch_list = self._generate_Sparse_batch_data_for_distributed_parts(X_trainA, X_trainB, Y_train, self.batch_size)
            self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
            print('Generation done.')
        else:
            raise Exception("[fit model] No proper entry for batch data generation. Check the data_tag or the fit function.")
        
        """ Train Model """
        self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_list, instances_count)


    def fit_model_secure_distributed_input(self, X_batch_listA, X_batch_listB, y_batch_list, instances_count):
        
        self.n_iteration = 0
        self.loss_history = []
        test = 0

        # print("[CHECK] weight: ", self.weightA, self.weightB)
        
        #### Secret share model
        # print("secret sharing model...")
        wa1, wa2 = self.secret_share_vector_plaintext(self.weightA)
        # print("wa1+wa2: ", wa1 + wa2)
        wb1, wb2 = self.secret_share_vector_plaintext(self.weightB)

        ############################
        # import time
        # filename = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        # self.logname = "CAESAR_" + filename + ".txt"
        # file = open(self.logname, mode='w+') #  写入记录
        # time_start_training = time.time()
        ############################
        
        # print("training model...")
        while self.n_iteration < self.max_iter:
            time_start_training = time.time()
            loss_list = []
            batch_labels = None
            # distributed
            test = 0
            for batch_dataA, batch_dataB, batch_labels, batch_num in zip(X_batch_listA, X_batch_listB, y_batch_list, self.batch_num):
                batch_labels = batch_labels.reshape(-1, 1)
                # print("batch ", test)

                ############################
                # file.write("batch " + str(test) + "\n")
                ############################

                test += 1
                ##################### secure forward #####################
                # print("forwarding...")
                # print("wa1,wa2 type: ", type(wa1), type(wa2))
                self.secure_distributed_cal_z(X = batch_dataA, w1 = wa1, w2 = wa2, party = "A")
                self.secure_distributed_cal_z(X = batch_dataB, w1 = wb1, w2 = wb2, party = "B")
                
                ################## 修改了这里的za1, 做了转置, 原来的是1*40没有对齐，现在是40*1
                self.za = self.za1.T + self.za2_1 + self.zb1_1
                self.zb = self.zb2.T + self.za2_2 + self.zb1_2
                # print("[CHECK] w dot x: ", self.za + self.zb)
                # print(type(self.za))
                # print(type(self.zb))

                encrypt_za = self.cipher.recursive_encrypt(self.za)
                # encrypt_zb = np.asarray(self.pool.map(self.cipher.recursive_encrypt, self.zb))
                # print("type encrypt_zb: ", type(encrypt_zb))

                # wx encrypt
                self.encrypt_wx = self.zb + encrypt_za
                # sigmoid
                self.encrypted_sigmoid_wx = self._compute_sigmoid(self.encrypt_wx)
                # error
                self.encrypted_error = (self.encrypted_sigmoid_wx - batch_labels).T
                # print("batch_labels shape: ", batch_labels.shape)
                
                # print("[CHECK] error: ", self.encrypted_error)
                
                # print("self.encrypted_sigmoid_wx[0] type: ", type(self.encrypted_sigmoid_wx[0]))
                # yb_s, ya_s = self.secret_share_vector_plaintext(self.encrypted_sigmoid_wx)
                yb_s, ya_s = self.secret_share_vector(self.encrypted_sigmoid_wx)
                # print("yb_s type: ", type(yb_s))
                error_b = yb_s - batch_labels
                error_a = ya_s

                # print("yb_s shape: ", yb_s.shape)
                # print("batch_labels shape: ", batch_labels.shape)

                # import sys
                # sys.exit(0)
                ########

                ########################## secure backward ####################
                ## Guest(B) backward 
                # print("backwarding...")
                # gradient = np.dot(error_1_n.T, batch_dataB)
                # encrypt_gb = self.encrypted_error.dot(batch_dataB) * (1 / batch_num)
                assert(self.encrypted_error.shape[1] == batch_dataB.shape[0])
                encrypt_gb = np.dot(self.encrypted_error, batch_dataB) * (1 / batch_num)
                # print("encrypt_gb shape: ", encrypt_gb.shape)
                
                # gb2, gb1 = self.secret_share_vector_plaintext(encrypt_gb) # 前面一个返回值是留在本方的值, 后面一个是share给对方的值
                gb2, gb1 = self.secret_share_vector(encrypt_gb)
                # print("gb1,gb2 shape: ", gb1.shape, gb2.shape)
                # print("wb1,wb2 shape: ", wb1.shape, wb2.shape)

                ## Host(A) backward
                error_1_n = error_b * (1 / batch_num)
                ga2_2, ga2_1 = self.secure_Matrix_Multiplication(batch_dataA, error_1_n, stage = "backward")
                assert(self.encrypted_error.shape[1] == batch_dataB.shape[0])
                ga = np.dot(error_a.T, batch_dataA) * (1 / batch_num)

                ga_new = ga + ga2_1

                ########################## compute loss #######################
                # print("computing loss ...")
                batch_loss = self.secure_distributed_compute_loss_cross_entropy(label = batch_labels, batch_num = batch_num)
                loss_list.append(batch_loss)
                
                ########################## update model #######################
                # print("update model ...")
                wa1 = wa1 - self.alpha * ga_new - self.lambda_para * self.alpha * wa1 / batch_num
                wa2 = wa2 - self.alpha * ga2_2 - self.lambda_para * self.alpha * wa2 / batch_num
                wb1 = wb1 - self.alpha * gb1 - self.lambda_para * self.alpha * wb1 / batch_num
                wb2 = wb2 - self.alpha * gb2 - self.lambda_para * self.alpha * wb2 / batch_num

                # l2-penalty
                # self.model_weights = self.model_weights - self.alpha * gradient - self.lambda_para * self.alpha * self.model_weights / batch_num
                # time_end_training = time.time()
                # print('batch cost: ',time_end_training-time_start_training,'s')

            # 打乱数据集的batch
            X_batch_listA, X_batch_listB, y_batch_list = self.shuffle_distributed_data(X_batch_listA, 
                                X_batch_listB, y_batch_list)
            
            ## 计算 sum loss
            loss = np.sum(loss_list) / instances_count
            loss_decrypt = self.cipher.recursive_decrypt(loss)
            # print("\rIteration {}, batch sum loss: {}".format(self.n_iteration, loss_decrypt))
            # # self.loss_history.append(loss_decrypt)
            
            ############################
            time_end_training = time.time()
            # c
            # print(" Time: " + str(time_end_training-time_start_training) + "s")
            # file.write("Time: " + str(time_end_training-time_start_training) + "s\n")
            # file.write("\nEpoch {}, batch sum loss: {}".format(self.n_iteration, loss_decrypt))

            ############################


            ## 判断是否停止
            self.is_converged = self.check_converge_by_loss(loss_decrypt)
            if self.is_converged:
                # self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
                if self.ratio is not None: 
                    # self.weightA = self.cipher.recursive_decrypt(wa1 + wa2)
                    # self.weightB = self.cipher.recursive_decrypt(wb1 + wb2)

                    self.weightA = wa1 + wa2
                    self.weightB = wb1 + wb2

                    self.model_weights = np.hstack((self.weightA, self.weightB))
                    print("\nEpoch {}, batch sum loss: {}".format(self.n_iteration, loss_decrypt), end='')
                    # print("self.model_weights: ", self.model_weights)
                break

            self.n_iteration += 1

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

    def _generate_batch_data_for_distributed_parts(self, X1, X2, y, batch_size):
        '''
        输入的数据就是两部分的
        目的是将这两部分横向ID对齐的数据 划分成一个个batch (可用于实验中的分别采样输入数据)
        ratio: 决定划分到含有标签的一方的数据的比例,对划分的数量下取整,例如 0.8 * 63 -> 50
        '''
        print("batch data generating...")
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
            self.model_weights = self.model_weights.reshape(-1, 1)
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

        file = open(self.logname, mode='a+') #  写入记录
        file.write("\nPredict precision: {}".format(rate))
        file.close()



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
        # batch 数据生成
        X_batch_listA, X_batch_listB, y_batch_list = self._generate_batch_data_for_distributed_parts(X_train1, X_train2, 
                                                                                        Y_train, self.batch_size)
        self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
        
        for i in range(len(label_lst)):
            # 转换标签值为二分类标签值
            pos_label = label_lst[i]                                        # 选定正样本的标签
            print("Label: ", pos_label)

            def label_reset_OVR(arr):
                """ 依次将标签i设置为正样本, 其他为负样本 """
                # global pos_label
                return np.where(arr == pos_label, 1, 0)
            
            y_batch_list = list(map(label_reset_OVR, y_batch_list))
            print("y_batch_list ok.")
            
            # Y_train_new = np.where(Y_train == pos_label, 1, 0)              # 满足条件则为正样本1，否则为负样本0
            # Y_test_new = np.where(Y_test == pos_label, 1, 0)
            # print(Y_train_new)
            self.fit_model_secure_distributed_input(X_batch_listA, X_batch_listB, y_batch_list, instances_count)
            
            print("fit_model done.")
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
        print(y_predict)
        # 模型预测准确率
        score = 0
        for i in range(len(y_predict)):
            if y_predict[i] == Y_test[i]:
                score += 1
            else:
                pass
        rate = score / len(y_predict)
        print("Predict precision: ", rate)






def read_distributed_data():
    from sklearn.datasets import load_svmlight_file
    import os

    dataset_file_name = 'DailySports'  
    train_file_name = 'DailySports_train.txt' 
    test_file_name = 'DailySports_test.txt'
    main_path = PATH_DATA

    # dataset_file_name = 'a6a'
    # train_file_name = 'a6a.txt'
    # test_file_name = 'a6a.t'
    # main_path = '/Users/zbz/data/'
    train_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, train_file_name))
    test_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, test_file_name))
    X_train = train_data[0].todense().A
    X_test = test_data[0].todense().A
    print("X_train type: ", type(X_train)) # 1000 * 60
    print("X_train shape: ", X_train.shape)
    print("X_test type: ", type(X_test)) # 1000 * 60
    print("X_test shape: ", X_test.shape)

    Y_train = train_data[1].astype(int)
    # X_test = test_data[0]
    Y_test = test_data[1].astype(int)
    
    # print(Y_train[0]) # 1000 * 1

    # 判断标签是(1;-1)还是 (1;0), 将-1的标签全部转化成0标签
    # if -1 in Y_train:  
    #     Y_train[Y_train == -1] = 0
    #     Y_test[Y_test == -1] = 0

    print("processing dataset...")
    Y_train[Y_train != 1] = 0
    Y_test[Y_test != 1] = 0
    # print(Y_train)
    # print(Y_test)

    k = X_train.shape[1]
    partition = 3/10
    k1 = np.floor(k * partition).astype(int)
    X_train1, X_train2 = X_train[:,0:k1], X_train[:,k1:]

    k = X_test.shape[1]
    partition = 3/10
    k1 = np.floor(k * partition).astype(int)
    X_test1, X_test2 = X_test[:,0:k1], X_test[:,k1:]

    print("X_train1 and X_train2 shape: ", X_train1.shape, X_train2.shape)

    # import sys
    # sys.exit()

    # #a6a a7a
    # X_train = X_train.todense().A
    # X_train = np.hstack( (X_train, np.zeros(X_train.shape[0]).reshape(-1, 1)) )
    # return ss.fit_transform(X_train), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array

    # #splice
    # return ss.fit_transform(X_train.todense().A), Y_train, ss.fit_transform(X_test.todense().A), Y_test # matrix转array
    # # return X_train.todense().A, Y_train, X_test.todense().A, Y_test # matrix转array
    
    # return X_train, Y_train, X_test, Y_test # matrix转array
    return X_train1, X_train2, Y_train, X_test1, X_test2, Y_test 

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




if __name__ == "__main__":
    # print("Hi.")
    # import sys
    # sys.exit(0)
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
    #                 max_iter = 2000, alpha = 0.0001, eps = 1e-6, penalty = None, lambda_para = 1, data_tag=None)  
                    # breast_cancer: 0.9649122807017544
                    # splice: 0.8482758620689655
                    # splice 集中 0.9062068965517242
    # 纵向划分分布式
    LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 128, 
                    max_iter = 200, alpha = 0.0001, eps = 1e-7, ratio = 0.7, penalty = None, lambda_para = 1, data_tag = None)
                    # splice 分布式 0.9062068965517242
    # LogisticRegressionModel = LogisticRegression(weight_vector = weight_vector, batch_size = 20, 
    #                 max_iter = 600, alpha = 0.0001, eps = 1e-6, ratio = 0.7, penalty = None, lambda_para = 1, data_tag = None)

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
    # indice_littleside = X_train1.shape[1]
    # LogisticRegressionModel.fit_model_distributed_input(X_train1, X_train2, Y_train, X_train1.shape[0], indice_littleside)

    # 纵向分布保护隐私的分布式
    indice_littleside = X_train1.shape[1]
    # LogisticRegressionModel.fit_model_secure_distributed_input(X_train1, X_train2, Y_train, X_train1.shape[0], indice_littleside)
    """ 多分类 """
    LogisticRegressionModel.OVRClassifier(X_train1, X_train2, X_test1, X_test2, Y_train, Y_test)

    time_end = time.time()
    print("SecureMLModel.train_time_account: ", LogisticRegressionModel.train_time_account)
    print('Total time cost: ', time_end-time_start + LogisticRegressionModel.train_time_account,'s')

    # plt.plot(LogisticRegressionModel.loss_history)
    # plt.show()

    ########## 二分类测试 ##########
    # 理想集中和伪分布式
    # LogisticRegressionModel.predict(X_test, y_test)
    # 纵向划分分布式
    # LogisticRegressionModel.predict_distributed(X_test1, X_test2, Y_test)