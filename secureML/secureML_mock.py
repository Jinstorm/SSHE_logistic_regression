import numpy as np
import time
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from multiprocessing import Pool, Process, Queue
# import multiprocessing

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_pardir = os.path.join(dir_path, os.pardir)
# abs_parpardir = os.path.join(abs_pardir, os.pardir)
sys.path.append(abs_pardir)
# print(abs_parpardir)
from paillierm.encrypt import PaillierEncrypt
from paillierm.fixedpoint import FixedPointEndec
from paillierm.utils import urand_tensor

PATH_DATA = '../data/' # '../../data/'
# flag = '' # sketch or raw

class SecureML:
    """
    SecureML Implementation
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
        self.data_tag = data_tag # 输入数据的格式 (目前支持两种格式: sparse和dense)

        # 加密部分的初始化
        self.cipher = PaillierEncrypt() # Paillier初始化
        self.cipher.generate_key()  # Paillier生成公私钥
        self.fixedpoint_encoder = FixedPointEndec(n = 1e10) # 加密前的定点化编码器初始化

        # 进程池
        # self.pool = Pool()

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


    def secret_share_vector_plaintext(self, share_target):
        '''
        Desc: 秘密分享(输入的share_target是明文)
        '''
         # 生成本地share向量
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
        share = share_target - tmp
        return tmp, share # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share


    def secure_distributed_compute_loss_cross_entropy(self, label, Y_predictA, Y_predictB, batch_num):
        """
        Input
        -----
        label, Y_predictA(wxa), Y_predictB(wxb), batch_num

        Desc
        -----
            Use Taylor series expand log loss:
            Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
            Then loss' = - (1/N)*∑ ( log(1/2) - 1/2*wx + ywx -1/8(wx)^2 )
        """
        # self.encrypted_wx = self.wx_self_A + self.wx_self_B
        wx = Y_predictA + Y_predictB
        # print("wx: ", wx)
        # print("wx shape: ", wx.shape)
        # print("label: ", label)
        # print("label shape: ", label.shape)

        # import sys
        # sys.exit()

        half_wx = -0.5 * wx
        assert(wx.shape[0] == label.shape[0])
        ywx = wx * label

        # wx_square = (self.za * self.za + 2 * self.za * self.zb + self.zb * self.zb) * -0.125
        wx_square = (Y_predictA * Y_predictA + 2 * Y_predictA * Y_predictB + Y_predictB * Y_predictB) * -0.125
        # wx_square = np.dot(self.wx_self.T, self.wx_self) * -0.125 # 这里后续要修改，两方平方，有交叉项
        # assert all(wx_square == wx_square2)  # 数组比较的返回值为: 类似[True False False]

        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )
        return loss
    
    def secretSharing_Data_and_Labels(self, data_matrixA, data_matrixB, Y_train):
        '''
        将数据X和标签Y, 分享到两方.
        '''
        local_dataA, share_dataA = self.secret_share_vector_plaintext(data_matrixA)
        local_dataB, share_dataB = self.secret_share_vector_plaintext(data_matrixB)
        local_Y, share_Y = self.secret_share_vector_plaintext(Y_train)
        

        self.local_matrix_A = np.hstack((local_dataA, share_dataB))
        self.local_matrix_B = np.hstack((share_dataA, local_dataB))
        self.Y_A = local_Y
        self.Y_B = share_Y
        assert(self.local_matrix_A.shape == self.local_matrix_B.shape)
        print("Sharing raw data: \033[32mOK\033[0m")

    def fit_model_secure_distributed_input(self, X_trainA, X_trainB, Y_train, instances_count, feature_count, indice_littleside):
        # indice_littleside 用于划分权重, 得到特征数值较小的那一部分的权重-或者左侧 默认X1一侧
        # mini-batch 数据集处理
        print("ratio: ", self.ratio)
        self.indice = indice_littleside

        # generate shared data and labels for two parties
        self.secretSharing_Data_and_Labels(X_trainA, X_trainB, Y_train)
        # label: self.Y_A self.Y_B
        # data: self.local_matrix_A self.local_matrix_B

        # split the model weight according to data distribution
        self.weightA = self.model_weights
        self.weightB = self.model_weights

        # generate triples: U V Z V' Z'
        # print("len n/|B|: ", len(self.batch_num))
        # print(self.batch_num)
        # import sys
        # sys.exit()
        import math
        t = int(math.ceil(instances_count/self.batch_size))
        print("t: ", t)
        self.generate_UVZV_Z_multTriplets_beaver_triplets(instances_count, feature_count, 
                                                          t, self.batch_size)
        # Mask X0 X1 and reconstruct E
        E0 = self.local_matrix_A - self.U0
        E1 = self.local_matrix_B - self.U1
        E = self.reconstruct(E0, E1)

        # generate batch data:
        X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)
        # X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, Zp1_batch_list, Zp2_batch_list = self._generate_batch_data_and_triples(E, self.batch_size)
        # 这些batch data可以在过程中计算得到: Z_batch_list, Z_p_batch_list (算了一起生成吧)

        self.n_iteration = 0
        self.loss_history = []
        test = 0

        print("[CHECK] weight: ", self.weightA, self.weightB)
        self.weightA = self.weightA.reshape(-1, 1)
        self.weightB = self.weightB.reshape(-1, 1)

        ############################
        import time
        filename = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        name = filename + ".txt"
        self.file = open(name, mode='w+') #  写入记录
        # time_start_training = time.time()
        ############################
        
        print("[Hint] Training model...")
        while self.n_iteration < self.max_iter:
            time_start_training = time.time()
            loss_list = []
            batch_label_A = None
            batch_label_B = None
            # distributed
            test = 0
            for batch_dataA, batch_dataB, batch_label_A, batch_label_B, batch_E, batch_Z0, batch_Z1, batch_U0, batch_U1, batch_num in zip(X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, 
                                                                         E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list, self.batch_num):
                ############################
                # file.write("batch " + str(test) + "\n")
                ############################
                test += 1
                
                batch_label_A = batch_label_A.reshape(-1, 1)
                batch_label_B = batch_label_B.reshape(-1, 1)

                j = 0
                # ?
                # print("[CHECK] self.V0 self.V1 shape: ", self.V0.shape, self.V1.shape)
                # import sys
                # sys.exit(0)
                batch_F0 = self.weightA - self.V0[:,j].reshape(-1, 1)
                batch_F1 = self.weightB - self.V1[:,j].reshape(-1, 1)
                batch_F = self.reconstruct(batch_F0, batch_F1)
                # print("batch_F shape: ", batch_F.shape, batch_F0.shape, batch_F1.shape, self.weightA.shape, self.weightB.shape, self.V0[:,j].reshape(-1, 1).shape)

                # compute the predict Y*
                # ???????????????????????????????????????????????? 按理说加起来应该等于Xw的, 检查一下原理公式和代码
                Y_predictA = np.dot(batch_dataA, batch_F) + np.dot(batch_E, self.weightA) + batch_Z0[:,j].reshape(-1, 1)
                Y_predictB = np.dot(batch_dataB, batch_F) + np.dot(batch_E, self.weightB) + batch_Z1[:,j].reshape(-1, 1) + -1 * np.dot(batch_E, batch_F)
                # print("shape: ", np.dot(batch_dataA, batch_F).shape, np.dot(batch_E, self.weightA).shape, batch_Z0[:,j].reshape(-1, 1).shape)
                # print("shape: ", np.dot(batch_dataB, batch_F).shape, np.dot(batch_E, self.weightB).shape, batch_Z1[:,j].reshape(-1, 1).shape, np.dot(batch_E, batch_F).shape)
                Y_predictA = self._compute_sigmoid(Y_predictA)
                Y_predictB = self._compute_sigmoid_dual_distributed(Y_predictB)


                # compute the difference
                # print("Y_predictA shape: ", Y_predictA.shape)
                # print("Y_predictB shape: ", Y_predictB.shape)
                # print("batch_label_A shape: ", batch_label_A.shape)

                batch_D0 = Y_predictA - batch_label_A
                batch_D1 = Y_predictB - batch_label_B

                # print("batch_D0 shape: ", batch_D0.shape)
                # print("batch_D1 shape: ", batch_D1.shape)

                # import sys
                # sys.exit()

                if len(batch_D0) != self.batch_size:
                    # 最后一个不足一个batchsize的数据片
                    end = len(batch_D0)

                    batch_Z_ = np.dot((batch_U0+batch_U1).T, self.V_[0:end,j].reshape(-1, 1))
                    batch_Z0_, batch_Z1_ = self.secret_share_vector_plaintext(batch_Z_)

                    batch_Fp0 = batch_D0 - self.V0_[0:end,j].reshape(-1, 1)
                    batch_Fp1 = batch_D1 - self.V1_[0:end,j].reshape(-1, 1)
                    batch_Fp = self.reconstruct(batch_Fp0, batch_Fp1)

                    # delta0 = np.dot(batch_dataA.T, batch_Fp) + np.dot(batch_E.T, batch_D0) + np.dot(batch_U0.T, self.V0_[0:end,j]).reshape(-1, 1)
                    # delta1 = np.dot(batch_dataB.T, batch_Fp) + np.dot(batch_E.T, batch_D1) + np.dot(batch_U1.T, self.V1_[0:end,j]).reshape(-1, 1) + -1 * np.dot(batch_E.T, batch_Fp)
                    delta0 = np.dot(batch_dataA.T, batch_Fp) + np.dot(batch_E.T, batch_D0) + batch_Z0_
                    delta1 = np.dot(batch_dataB.T, batch_Fp) + np.dot(batch_E.T, batch_D1) + batch_Z1_ + -1 * np.dot(batch_E.T, batch_Fp)
                    
                else:
                    batch_Z_ = np.dot((batch_U0+batch_U1).T, self.V_[:,j].reshape(-1, 1))
                    batch_Z0_, batch_Z1_ = self.secret_share_vector_plaintext(batch_Z_)

                    batch_Fp0 = batch_D0 - self.V0_[:,j].reshape(-1, 1)
                    batch_Fp1 = batch_D1 - self.V1_[:,j].reshape(-1, 1)
                    batch_Fp = self.reconstruct(batch_Fp0, batch_Fp1)
                    # print("batch_Fp shape: ", batch_Fp.shape)

                    # print("shape: ", np.dot(batch_dataA.T, batch_Fp).shape, np.dot(batch_E.T, batch_D0).shape, np.dot(batch_U0.T, self.V0_[:,j]).reshape(-1, 1).shape)
                    # delta0 = np.dot(batch_dataA.T, batch_Fp) + np.dot(batch_E.T, batch_D0) + np.dot(batch_U0.T, self.V0_[:,j].reshape(-1, 1))
                    # print("value: ", np.dot(batch_dataA.T, batch_Fp),np.dot(batch_E.T, batch_D0),np.dot(batch_U0.T, self.V0_[:,j].reshape(-1, 1)))
                    # delta1 = np.dot(batch_dataB.T, batch_Fp) + np.dot(batch_E.T, batch_D1) + np.dot(batch_U1.T, self.V1_[:,j].reshape(-1, 1)) + -1 * np.dot(batch_E.T, batch_Fp)
                    
                    delta0 = np.dot(batch_dataA.T, batch_Fp) + np.dot(batch_E.T, batch_D0) + batch_Z0_
                    delta1 = np.dot(batch_dataB.T, batch_Fp) + np.dot(batch_E.T, batch_D1) + batch_Z1_ + -1 * np.dot(batch_E.T, batch_Fp)
                
                # truncates
                # ......

                # update
                # print("value: ", self.weightA, (self.alpha / batch_num * (delta0)))
                # print("shape: ", self.weightA.shape, (self.alpha / batch_num * (delta0)).shape)
                self.weightA = self.weightA - self.alpha / batch_num * (delta0)
                self.weightB = self.weightB - self.alpha / batch_num * (delta1)
                
                # print("weight: ", self.weightA + self.weightB)
                # import sys
                # sys.exit()

                j = j + 1
                # print()

                ########################## compute loss #######################
                # print("computing loss ...")
                batch_loss = self.secure_distributed_compute_loss_cross_entropy(label = batch_label_A + batch_label_B, 
                                                                        Y_predictA=Y_predictA, Y_predictB=Y_predictB, batch_num = batch_num)
                loss_list.append(batch_loss)

            # # 打乱数据集的batch
            # X_batch_listA, X_batch_listB, y_batch_list = self.shuffle_distributed_data(X_batch_listA, 
            #                     X_batch_listB, y_batch_list)
            
            ## 计算 sum loss
            loss = np.sum(loss_list) / instances_count
            print("\rIteration {}, batch sum loss: {}".format(self.n_iteration, loss))
            # self.loss_history.append(loss_decrypt)
            
            ############################
            time_end_training = time.time()
            # print('time cost: ',time_end_training-time_start_training,'s')
            self.file.write("Time: " + str(time_end_training-time_start_training) + "s\n")

            # file.write("loss shape: " + str(loss.shape) + "\n")
            self.file.write("\rIteration {}, batch sum loss: {}".format(self.n_iteration, loss))
            # self.file.close()
            ############################

            # import sys
            # sys.exit(0)


            ## 判断是否停止
            self.is_converged = self.check_converge_by_loss(loss)
            if self.is_converged:
                # self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
                if self.ratio is not None: 
                    # self.weightA = self.cipher.recursive_decrypt(wa1 + wa2)
                    # self.weightB = self.cipher.recursive_decrypt(wb1 + wb2)

                    # self.weightA = wa1 + wa2
                    # self.weightB = wb1 + wb2

                    self.model_weights = self.weightA + self.weightB
                    print("self.model_weights: ", self.model_weights)
                break

            self.n_iteration += 1


    def reconstruct(self, Ei, Ei_):
        E = Ei + Ei_ # 两方都各自重建E
        return E

    def generate_UVZV_Z_multTriplets_beaver_triplets(self, n, d, t, B):
        """
        Generate beaver_triplets and ss to two parties A and B. (Offline phase)

        Parameters
        ---------
        `X` - `n*d`input data matrix
        n: 样本数
        d: 总特征维度
        t: 每个epoch对应的batch数, t = n/B 上取整 np.seiling
        B: batch size 大小

        Return
        ---------
        `U0`, `U1`, `V0`, `V1`, `V0_`, `V1_`, `Z0`, `Z1`, `Z0_`, `Z1_`; _0 for Party 0 and _1 for Party 1.
        """
        U = np.random.rand(n, d)
        V = np.random.rand(d, t)
        self.V_ = np.random.rand(B, t)
        self.U0, self.U1 = self.secret_share_vector_plaintext(U)
        self.V0, self.V1 = self.secret_share_vector_plaintext(V)
        self.V0_, self.V1_ = self.secret_share_vector_plaintext(self.V_)


        self.Z = np.dot(U, V) # 按照下面两行写, 乘法缺项
        # self.Z0 = np.dot(self.U0, self.V0)
        # self.Z1 = np.dot(self.U1, self.V1)
        self.Z0, self.Z1 = self.secret_share_vector_plaintext(self.Z)

        # 这里Z_必须在训练中生成, 因为Z_对应的是每个的batch, 而不是整个数据集, Z_的维度: (|B|, t)
        # 注意遇到某个列不足|B|的长度时, Z_的生成需要注意维度: 此时Z
        # self.Z0_ = np.dot(self.U0.transpose(), self.V0_)
        # self.Z1_ = np.dot(self.U0.transpose(), self.V1_)

        # return U0, U1, V0, V1, Z0, Z1, V0_, V1_, Z0_, Z1_
        # 参考FATE或

    """
    TODO
    1 分享数据, 初始化模型参数.
    2 生成batch data, 且需要A和B的batch是对应的
    3 生成乘法三元组(简化版实现), U0 U1 V0 V1 Z0 Z1 V0' V1' Z0' Z1'

    E: n*d
    F: d*t

    TODO:目前修改到: generate batch 要加上E, 还有计算Y*预测值

    开始训练:
    Ei = Xi - Ui
    重建 E
    for
        两边都从第一个X_batch Y_batch开始训练, 
        Fi = wi - Vi
        重建F
        Y* = -i*EBj * Fi + XBi * Fi + EB * wi + ZBi
        DBi = Y*Bi - YBi

        F'Bi = DBi - V'Bi
        重建F'

        delta = i * EBT * F'B + XTBi * F'j + EBT * DBi + Z'Bi
        截断
        wi = wi - alpha/B (delta)
    重建w

    Now: 
    修改_generate_batch_data_for_localparts
    循环加入V取第j列的操作(需要核对第j列是否是第j个batch)
    F
    reconstruction
    Y...

    现在的问题: wx出来的值不对 和 Y 差的远了

    """
    

    def _generate_batch_data_and_triples(self, E, batch_size):
        # for two parties in secureML model to generate the batches
        # E X Y (V V',校对在迭代过程中,列序号能对应batch序号即可) Z Z'
        X_batch_listA = []
        X_batch_listB = []
        y_batch_listA = []
        y_batch_listB = []
        E_batch_list = []
        Z0_batch_list = []
        Z1_batch_list = []
        U0_batch_list = []
        U1_batch_list = []
        # Z_p0_batch_list = []
        # Z_p1_batch_list = []
        # self.indice = math.floor(ratio * X.shape[1]) # 纵向划分数据集，位于label一侧的特征数量
        
        for i in range(len(self.Y_A) // batch_size):
            # X_tmpA = X1[i * batch_size : i * batch_size + batch_size, :]
            X_batch_listA.append(self.local_matrix_A[i * batch_size : i * batch_size + batch_size, :])
            X_batch_listB.append(self.local_matrix_B[i * batch_size : i * batch_size + batch_size, :])
            y_batch_listA.append(self.Y_A[i * batch_size : i * batch_size + batch_size])
            y_batch_listB.append(self.Y_B[i * batch_size : i * batch_size + batch_size])
            
            # E, Z0, Z1, Z'0, Z'1, batch_num
            E_batch_list.append(E[i * batch_size : i * batch_size + batch_size])
            
            Z0_batch_list.append(self.Z0[i * batch_size : i * batch_size + batch_size])
            Z1_batch_list.append(self.Z1[i * batch_size : i * batch_size + batch_size])
            U0_batch_list.append(self.U0[i * batch_size : i * batch_size + batch_size])
            U1_batch_list.append(self.U1[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)

        if (len(self.Y_A) % batch_size > 0):
            # X_tmpA, X_tmpB = np.hsplit(X[len(y) // batch_size * batch_size:, :], [self.indice])
            # X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            X_batch_listA.append(self.local_matrix_A[len(self.Y_A) // batch_size * batch_size:, :])
            X_batch_listB.append(self.local_matrix_B[len(self.Y_A) // batch_size * batch_size:, :])
            y_batch_listA.append(self.Y_A[len(self.Y_A) // batch_size * batch_size:])
            y_batch_listB.append(self.Y_B[len(self.Y_A) // batch_size * batch_size:])
            
            # E, Z0, Z1, Z'0, Z'1, batch_num
            E_batch_list.append(E[len(self.Y_A) // batch_size * batch_size:])

            Z0_batch_list.append(self.Z0[len(self.Y_A) // batch_size * batch_size:])
            Z1_batch_list.append(self.Z1[len(self.Y_A) // batch_size * batch_size:])
            U0_batch_list.append(self.U0[len(self.Y_A) // batch_size * batch_size:])
            U1_batch_list.append(self.U1[len(self.Y_A) // batch_size * batch_size:])
            self.batch_num.append(len(self.Y_A) % batch_size)

        print("Batch data generation: \033[32mOK\033[0m")
        return X_batch_listA, X_batch_listB, y_batch_listA, y_batch_listB, E_batch_list, Z0_batch_list, Z1_batch_list, U0_batch_list, U1_batch_list # listA——持有label一侧，较多样本; listB——无label一侧
    
    
    def predict_distributed(self, x_test1, x_test2, y_test):
        # z = np.dot(x_test, self.model_weights.T)
        # z = x_test.dot(self.model_weights.T)
        # z = self._cal_z()
        x_test = np.hstack((x_test1, x_test2))
        if self.data_tag == 'sparse':
            z = x_test.dot(self.model_weights.T) # np.dot(features, weights.T)
        elif self.data_tag == None:
            # SecureML
            z = np.dot(x_test, self.model_weights)

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

        self.file.write("Predict precision: {}".format(rate))
        self.file.close()

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

    print("X_train1 and 2 shape: ", X_train1.shape, X_train2.shape)

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
    Y_train[Y_train != 1] = 0
    Y_test[Y_test != 1] = 0
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

    dataset_file_name = 'DailySports/portion37_pminhash/sketch512/countsketch/'
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

    # Raw data
    # X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_data()
    # Sketch data
    X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_squeeze_data()

    print(X_train1.shape, X_train2.shape, X_train1.shape[1], X_train2.shape[1], Y_train.shape, X_test1.shape, Y_test.shape)

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
    SecureMLModel = SecureML(weight_vector = weight_vector, batch_size = 256, 
                    max_iter = 50, alpha = 0.001, eps = 1e-6, ratio = 0.7, penalty = None, lambda_para = 1, data_tag = None)
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
    SecureMLModel.fit_model_secure_distributed_input(X_train1, X_train2, Y_train, X_train1.shape[0], (X_train1.shape[1]+X_train2.shape[1]), indice_littleside)
    # SecureMLModel.fit_model_secure_2process(X_train1, X_train2, Y_train, X_train1.shape[0], indice_littleside)

    time_end = time.time()
    print('Total time cost: ', time_end-time_start,'s')

    # plt.plot(LogisticRegressionModel.loss_history)
    # plt.show()

    ########## 测试 ##########
    # 理想集中和伪分布式
    # LogisticRegressionModel.predict(X_test, y_test)
    # 纵向划分分布式
    SecureMLModel.predict_distributed(X_test1, X_test2, Y_test)