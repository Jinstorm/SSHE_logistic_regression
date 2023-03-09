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


    def secret_share_vector_plaintext(self, share_target, flag):
        '''
        Desc: 秘密分享(输入的share_target是明文)
        '''
         # 生成本地share向量
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
        share = share_target - tmp
        if flag == "A":
            self.comm_Queue_B.put(share)
            return tmp # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share
        elif flag =="B":
            self.comm_Queue_A.put(share)
            return tmp # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share
        else:
            return tmp, share # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share
        


    def secret_share_vector(self, share_target, flag):
        '''
        Desc: 秘密分享(输入的share_target是个加密的)
        '''
        # 生成本地share向量
        # print("share_target shape: ", share_target.shape)
        
        _pre = urand_tensor(q_field = self.fixedpoint_encoder.n, tensor = share_target)
        tmp = self.fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
        # print("pre shape: ", _pre.shape)
        share = share_target - tmp
        if flag == "A":
            self.comm_Queue_B.put(self.cipher.recursive_decrypt(share))
            return tmp # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share
        elif flag =="B":
            self.comm_Queue_A.put(self.cipher.recursive_decrypt(share))
            return tmp # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share
        else:
            return tmp, self.cipher.recursive_decrypt(share) # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share

    def secure_Matrix_Multiplication(self, matrix, vector, stage = None, flag = None):
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
        return self.secret_share_vector(mul_result, flag)

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
        self.za = self.comm_Queue_B.get()
        # self.comm_Queue_B.get()

        wx_square = (self.za * self.za + 2 * self.za * self.zb + self.zb * self.zb) * -0.125
        # wx_square = (2*self.wx_self_A * self.wx_self_B + self.wx_self_A * self.wx_self_A + self.wx_self_B * self.wx_self_B) * -0.125 
        # wx_square = np.dot(self.wx_self.T, self.wx_self) * -0.125 # 这里后续要修改，两方平方，有交叉项
        # wx_square2 = self.encrypted_wx * self.encrypted_wx * -0.125
        # assert all(wx_square == wx_square2)  # 数组比较的返回值为: 类似[True False False]

        loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5) )
        # loss = np.sum( (half_wx + ywx + wx_square) * (-1 / batch_num) )
        return loss


    def fit_model_secure_2process(self, X_trainA, X_trainB, Y_train, instances_count, indice_littleside):

        # print("ratio: ", self.ratio)
        # self.indice = indice_littleside # math.floor(self.ratio * ( X_trainA.shape[1]+X_trainB.shape[1] ) ) # 纵向划分数据集，位于label一侧的特征数量
        # if self.data_tag == None: 
        #     self.X_batch_listA, self.X_batch_listB, self.y_batch_list = self._generate_batch_data_for_distributed_parts(X_trainA, X_trainB, Y_train, self.batch_size)
        #     self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
        #     print(self.weightA.shape, self.weightB.shape)
        # else:
        #     raise Exception("[fit model] No proper entry for batch data generation. Check the data_tag or the fit function.")
            
        self.n_iteration = 1
        # self.loss_history = []
        # test = 0
        self.terminal = False


        ########## IPC ###########
        self.comm_Queue_A = Queue() # A party message box
        self.comm_Queue_B = Queue() # B party message box
        self.comm_Queue_C = Queue() # B party message box
        # import multiprocessing
        # m = multiprocessing.Manager()
        # arr = m.list()

        import time
        name = "sketch_time.txt"
        file = open(name, mode='w+') #  写入记录
        time_start_training = time.time()
        
        host_A = Process(target = self.fit_model_secure_Host_A,args=(X_trainA,) )
        guest_B = Process(target = self.fit_model_secure_Guest_B,args=(instances_count, X_trainB, Y_train, self.comm_Queue_C) )

        # 开始
        host_A.start()
        guest_B.start()
        
        # while self.comm_Queue_C.empty(): time.sleep(5)
        # self.model_weights = self.comm_Queue_C.get()

        
        guest_B.join()
        
        print("Done.")
        host_A.terminate()
        host_A.join()
        print("Done.")

        time_end_training = time.time()
        file.write("Total Train time: " + str(time_end_training-time_start_training) + "s\n")
        file.close()

        # print(arr)
        # self.model_weights = np.asarray(arr)
        # print("self.model_weights3: ", self.model_weights)
        
        # print("self.model_weights4: ", self.model_weights)

    def reconstruct(self, Ei, flag):
        if flag == "A":
            self.comm_Queue_B.put(Ei)
            Ei_ = self.comm_Queue_A.get()
        elif flag == "B":
            self.comm_Queue_A.put(Ei)
            Ei_ = self.comm_Queue_B.get()
        
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
        `U0`, `U1`, `V0`, `V1`, `Z0`, `Z1`; _0 for Party 0 and _1 for Party 1.
        """
        flag = "fuzz"
        U = np.random.rand(n, d)
        V = np.random.rand(d, t)
        V_ = np.random.rand(B, t)
        self.U0, self.U1 = self.secret_share_vector_plaintext(self, U, flag)
        self.V0, self.V1 = self.secret_share_vector_plaintext(self, V, flag)
        self.V0_, self.V1_ = self.secret_share_vector_plaintext(self, V_, flag)

        self.Z0 = np.dot(self.U0, self.V0)
        self.Z1 = np.dot(self.U1, self.V1)
        self.Z0_ = np.dot(self.U0.transpose(), self.V0_)
        self.Z1_ = np.dot(self.U0.transpose(), self.V1_)

        # return U0, U1, V0, V1, Z0, Z1, V0_, V1_, Z0_, Z1_
        # 参考FATE或
    
    def secretSharingTrainData_A(self, data_matrix):
        import time
        time_start = time.time()
        flag = "A"
        data_A1 = self.secret_share_vector(data_matrix, flag)
        print("A did SS, and waiting for get from B...")
        data_B2 = self.comm_Queue_A.get()
        print("Done.(A)")

        time_end = time.time()
        print("Total Comm time: " + str(time_end-time_start) + "s\n")

        self.local_matrix_A = np.hstack((data_A1, data_B2))
        print("shape local A: ", self.local_matrix_A.shape)
        
    def secretSharingTrainData_B(self, data_matrix):
        flag = "B"
        data_B1 = self.secret_share_vector(data_matrix, flag)
        print("B did SS, and waiting for get from A...")
        data_A2 = self.comm_Queue_B.get()
        print("Done.(B)")

        self.local_matrix_B = np.hstack((data_A2, data_B1))
        print("shape local B: ", self.local_matrix_B.shape)

        # 后续各自生成batch数据

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

    """
    # def forward_Host_A(self, X, wa1, wb1, party):
    #     self._cal_z(X, wa1, party = party, encrypt = "paillier") # self.za1
        
    #     # Matrix multiplication
    #     self.comm_Queue_B.put(wb1)
    #     wa2 = self.comm_Queue_A.get()
    #     if wa2 is None: 
    #         # convergence B处检查到已满足收敛条件
    #         print("convergence--wa1 shape: ", wa1.shape)
    #         print("convergence--wb1 shape: ", wb1.shape)
    #         self.comm_Queue_B.put(wa1)
    #         import time
    #         time.sleep(1)

    #         self.comm_Queue_B.put(wb1)
    #         self.terminal = True
    #         return

    #     self.za2_1 = self.secure_Matrix_Multiplication(X, wa2, stage = "forward", flag = "A")
    #     # B : get self.za2_2

    #     self.zb1_1 = self.comm_Queue_A.get()

    #     self.za = self.za1.T + self.za2_1 + self.zb1_1
    #     ## Line 14

    # def forward_Guest_B(self, X, wb2, wa2, party):
    #     self._cal_z(X, wb2, party = party, encrypt = "paillier") # self.zb2
        
    #     # Matrix multiplication
    #     self.comm_Queue_A.put(wa2)
    #     wb1 = self.comm_Queue_B.get()
    #     self.zb1_2 = self.secure_Matrix_Multiplication(X, wb1, stage = "forward", flag = "B")
    #     # A : get self.zb1_1

    #     self.za2_2 = self.comm_Queue_B.get()

    #     self.zb = self.zb2.T + self.za2_2 + self.zb1_2
    #     ## Line 14
    

    def fit_model_secure_Host_A(self, X_trainA):
        # SS generate wa1 wa2, 
        # Host A keeps wa1 and sent wa2 to Guest B
        # Host A get wb1 from B
        flag = "A"
        print("secret sharing data A...")
        self.secretSharingTrainData_A(X_trainA)

        # initiate weight A
        wa = np.zeros(self.local_matrix_A.shape[1]).reshape(1, -1)
        print("Weight A: ", wa, wa.shape)

        # reconstruct E
        E0 = self.local_matrix_A - self.U0
        E = self.reconstruct(E0, flag)

        # need to get Y_A
        y = self.comm_Queue_A.get()
        X_batch_listA, E_batch_list, Z_batch_list, y_batch_listA = self._generate_batch_data_for_localparts(self.local_matrix_A, y, E, self.batch_size)
        

        while self.n_iteration <= self.max_iter:
            # print("wa111 shape: ", wa1.shape)
            for batch_dataA, batch_E, batch_Z, batch_Y, batch_num in zip(X_batch_listA, E_batch_list, Z_batch_list, y_batch_listA, self.batch_num):
                # print("forward..")
                j = 0
                Fj0 = wa - self.V0[:,j]
                j = j + 1

                Fj = self.reconstruct(Fj0)
                
                Y_predict = np.dot(batch_dataA, Fj) + np.dot(batch_E, wa) + batch_Z

                # self.comm_Queue_B.put(self.cipher.recursive_encrypt(self.za))

                # ya_s = self.comm_Queue_A.get()
                # error_a = ya_s

                # # backward
                # # print("backward..")
                # ga = np.dot(error_a.T, batch_dataA) * (1 / batch_num)
                # gb1 = self.comm_Queue_A.get()

                # error_1_n = self.comm_Queue_A.get()
                # ga2_1 = self.secure_Matrix_Multiplication(batch_dataA, error_1_n, stage = "backward", flag = "A")
                
                # # compute loss
                # self.comm_Queue_B.put(self.za)

                # # update model
                # ga_new = ga + ga2_1
                # # print("ga_new: ", ga_new.shape)
                
                # wa1 = wa1 - self.alpha * ga_new - self.lambda_para * self.alpha * wa1 / batch_num
                # wb1 = wb1 - self.alpha * gb1 - self.lambda_para * self.alpha * wb1 / batch_num
                # # print("wa1,wb1: ", wa1,wb1)
                # # print("wa1 shape: ", wa1.shape)
                # # import sys
                # # sys.exit()
            
                # # self.comm_Queue_A.get()
            if self.terminal == True: 
                print("A return while")
                break
            


    def fit_model_secure_Guest_B(self, instances_count, X_trainB, Y_train, q):
        # Guest B keeps features and labels

        flag = "B"

        print("secret sharing data B...")
        self.secretSharingTrainData_B(X_trainB)

        # weight 初始化, 不需要通信, 两边分别随机初始化 1*d, 最后加起来重建即可.
        wb = np.zeros(self.local_matrix_B.shape[1]).reshape(1,-1)
        print("Weight B: ", wb, wb.shape)
        
        # reconstruct E
        E1 = self.local_matrix_B - self.U1
        E = self.reconstruct(E1, flag)

        # share Y_train
        y = self.secret_share_vector_plaintext(Y_train, flag)
        X_batch_listB, y_batch_listB = self._generate_batch_data_for_localparts(self.local_matrix_B, y, self.batch_size)
        
        # import time
        # filename = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        # name1 = filename + "_Epoch.txt"
        # file = open(name1, mode='w+') #  写入记录每个epoch的时间和loss
        # name2 = filename + "_Batch.txt"
        # file2 = open(name2, mode='w+') #  写入 some trash or useful words
        # file.write("sketch data." + "\n")
        
        while self.n_iteration <= self.max_iter:
            time_start_training = time.time()
            loss_list = []
            batch_labels = None
            i_batch = 0
            for batch_dataB, batch_labels, batch_num in zip(X_batch_listB, y_batch_listB, self.batch_num):
                j = 0
                Fj1 = wb - self.V1[:,j]
                j = j + 1

                self.reconstruct(Fj1)
        #         # file.write("batch ", i_batch)
        #         file2.write("batch " + str(i_batch))
        #         print("batch " + str(i_batch), end = '')
        #         print("forward..", end = '')

        #         batch_labels = batch_labels.reshape(-1, 1)
        #         self.forward_Guest_B(batch_dataB, wb2, wa2, party = "B")
        #         encrypt_za = self.comm_Queue_B.get()
        #         self.encrypt_wx = self.zb + encrypt_za
        #         self.encrypted_sigmoid_wx = self._compute_sigmoid(self.encrypt_wx)

        #         # self.backward_Guest_B(batch_labels, batch_num)
                
        #         # backward
        #         # print("backward..")
        #         yb_s = self.secret_share_vector(self.encrypted_sigmoid_wx, flag)
        #         error_b = yb_s - batch_labels

        #         self.encrypted_error = (self.encrypted_sigmoid_wx - batch_labels).T

        #         encrypt_gb = np.dot(self.encrypted_error, batch_dataB) * (1 / batch_num)
        #         gb2 = self.secret_share_vector(encrypt_gb, flag)         # 前面一个返回值是留在本方的值, 后面一个是share给对方的值
        #         error_1_n = error_b * (1 / batch_num)

        #         self.comm_Queue_A.put(error_1_n)
        #         ga2_2 = self.comm_Queue_B.get()

        #         # compute loss
        #         batch_loss = self.secure_distributed_compute_loss_cross_entropy(label = batch_labels, batch_num = batch_num)
        #         loss_list.append(batch_loss)

        #         # update model
        #         wa2 = wa2 - self.alpha * ga2_2 - self.lambda_para * self.alpha * wa2 / batch_num
        #         wb2 = wb2 - self.alpha * gb2 - self.lambda_para * self.alpha * wb2 / batch_num
        #         # print("wa2,wb2: ", wa2,wb2)

        #         # 对应一下get和put的数量和顺序
        #         time_end_training = time.time()
        #         file2.write('batch cost: ' + str(time_end_training-time_start_training)+'s')
        #         print("batch cost: " + str(time_end_training-time_start_training)+'s', end = '')

        #         i_batch += 1

        #     # 打乱数据集的batch
        #     # self.X_batch_listA, self.X_batch_listB, self.y_batch_list = self.shuffle_distributed_data(self.X_batch_listA, 
        #     #                     self.X_batch_listB, self.y_batch_list)
            
        #     # self.comm_Queue_A.put("\n")

        #     ## 计算 sum loss
        #     loss = np.sum(loss_list) / instances_count
        #     loss_decrypt = self.cipher.recursive_decrypt(loss)
        #     time_end_training = time.time()

        #     print("\nEpoch {}, batch sum loss: {}".format(self.n_iteration, loss_decrypt), end='')
        #     print(" Time: " + str(time_end_training-time_start_training) + "s")
        #     file.write("Time: " + str(time_end_training-time_start_training) + "s\n")
        #     # file.write("loss shape: " + str(loss.shape) + "\n")
        #     file.write("Epoch {}, batch sum loss: {}\n".format(self.n_iteration, loss_decrypt))
        #     file2.write("\nEpoch {}, batch sum loss: {}\n".format(self.n_iteration, loss_decrypt))
            
            
        #     self.is_converged = self.check_converge_by_loss(loss_decrypt)
        #     if self.is_converged or self.n_iteration == self.max_iter:
        #         # self.weightA, self.weightB = np.hsplit(self.model_weights, [self.indice]) # 权重向量是一个列向量，需要横向划分
        #         if self.ratio is not None: 
        #             # self.weightA = self.cipher.recursive_decrypt(wa1 + wa2)
        #             # self.weightB = self.cipher.recursive_decrypt(wb1 + wb2)
        #             self.comm_Queue_A.put(None)
                    
        #             self.wb1 = self.comm_Queue_B.get()
        #             self.wa1 = self.comm_Queue_B.get()
                    
        #             # print("wa1 shape: ", wa1.shape)
                    
        #             # print("wb1 shape: ", wb1.shape)

        #             self.weightA = self.wa1 + wa2
        #             self.weightB = self.wb1 + wb2
        #             print("wa1,wa2: ", self.wa1, wa2)

        #             self.model_weights = np.hstack((self.weightA, self.weightB))
        #             print("self.model_weights2: ", self.model_weights)
        #             self.comm_Queue_C.put(self.model_weights)
                    
        #         break

        #     self.n_iteration += 1
    

    def _generate_batch_data_for_localparts(self, X, y, batch_size):
        # for two parties in secureML model to generate the batches
        # E X Y (V V',校对在迭代过程中,列序号能对应batch序号即可) Z Z'
        X_batch_list = []
        y_batch_list = []

        for i in range(len(y) // batch_size):
            # X_tmpA = X1[i * batch_size : i * batch_size + batch_size, :]
            X_batch_list.append(X[i * batch_size : i * batch_size + batch_size, :])
            y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
            self.batch_num.append(batch_size)

        if (len(y) % batch_size > 0):
            # X_tmpA, X_tmpB = np.hsplit(X[len(y) // batch_size * batch_size:, :], [self.indice])
            # X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
            y_batch_list.append(y[len(y) // batch_size * batch_size:])
            self.batch_num.append(len(y) % batch_size)

        return X_batch_list, y_batch_list # listA——持有label一侧，较多样本; listB——无label一侧



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
    X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_data()
    # Sketch data
    # X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = read_distributed_squeeze_data()

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
                    max_iter = 200, alpha = 0.001, eps = 1e-6, ratio = 0.7, penalty = None, lambda_para = 1, data_tag = None)
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
    SecureMLModel.fit_model_secure_2process(X_train1, X_train2, Y_train, X_train1.shape[0], indice_littleside)

    time_end = time.time()
    print('Total time cost: ', time_end-time_start,'s')

    # plt.plot(LogisticRegressionModel.loss_history)
    # plt.show()

    ########## 测试 ##########
    # 理想集中和伪分布式
    # LogisticRegressionModel.predict(X_test, y_test)
    # 纵向划分分布式
    SecureMLModel.predict_distributed(X_test1, X_test2, Y_test)