import numpy as np
import time
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
# from multiprocessing import Pool, Process, Queue
# import multiprocessing

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
np.random.seed(100)
fixedpoint_encoder = FixedPointEndec(n = 1e10)

def test_sharing_cost():
    share_target = np.random.random((10))
    print(share_target.shape, share_target.size)
    
    _pre = urand_tensor(q_field = fixedpoint_encoder.n, tensor = share_target)
    tmp = fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
    # print("pre shape: ", _pre.shape)
    share = share_target - tmp
    print(share_target)
    print(share+tmp)
    if (share_target - (share+tmp) < 1e-6).all(): print("Vector Sharing: \033[32mPASS\033[0m")

def SS_matrix_with_distributed_2matrix():
    share_target = np.random.rand(6,5)
    share_t1, share_t2 = np.hsplit(share_target, [3]) # 按列拆分
    _pre1 = urand_tensor(q_field = fixedpoint_encoder.n, tensor = share_t1)
    _pre2 = urand_tensor(q_field = fixedpoint_encoder.n, tensor = share_t2)
    tmp1 = fixedpoint_encoder.decode(_pre1)
    tmp2 = fixedpoint_encoder.decode(_pre2)
    share1 = share_t1 - tmp1
    share2 = share_t2 - tmp2

    local_matrix1 = np.hstack((tmp1, share2))
    local_matrix2 = np.hstack((share1, tmp2))

    reconstruct = local_matrix1 + local_matrix2

    if (share_target - reconstruct < 1e-6).all(): print("Matrix Sharing: \033[32mPASS\033[0m")

def secret_share_vector_plaintext(share_target, flag):
    '''
    Desc: 秘密分享(输入的share_target是明文)
    '''
    # 生成本地share向量
    _pre = urand_tensor(q_field = fixedpoint_encoder.n, tensor = share_target)
    tmp = fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
    share = share_target - tmp
    return tmp, share # 返回的第一个参数是留在本方的share, 第二个参数是需要分享的share

def test_np_mul_SS(n, d, t):
    # 测试秘密分享后的矩阵，是否可以直接拿来做矩阵乘法
    flag = "fuzz"
    U = np.random.rand(n, d)
    V = np.random.rand(d, t)
    U0, U1 = secret_share_vector_plaintext(U, flag)
    V0, V1 = secret_share_vector_plaintext(V, flag)

    Z0 = np.dot(U0, V0)
    Z1 = np.dot(U1, V1)
    Z_original = np.dot(U,V)
    Z = Z0 + Z1
    if (np.abs((Z_original - (Z0+Z1))) < 1e-4).all(): print("Shared Matrix MUL: \033[32mPASS\033[0m")
    print(Z_original[0][0], Z[0][0])
    # print(Z_original, Z)

    print("Z0 Z1 shape: ", Z0.shape, Z1.shape)
    return Z0, Z1

def test_np_add_SS(n, d, t):
    # 测试秘密分享后的矩阵，是否可以直接拿来做矩阵乘法
    flag = "fuzz"
    U = np.random.rand(n, d)
    U0, U1 = secret_share_vector_plaintext(U, flag)

    U_ = U0 + U1
    if (np.abs(U - U_) < 1e-4).all(): print("Shared Matrix Add: \033[32mPASS\033[0m")
    print(U[0][0], U_[0][0])
    # print(Z_original, Z)

    print("U U_ shape: ", U.shape, U_.shape)
    return U, U_


def boolean_ss():
    data = np.random.randint(0,2,size = (1,10))
    b = np.random.randint(0,2,size = (1,10))
    print(data,b)
    c = np.logical_xor(data, b)

    print(np.logical_xor(c, b))

if __name__ == '__main__':
    # test_sharing_cost()
    # SS_matrix_with_distributed_2matrix()
    # test_np_mul_SS(3,4,5)
    # test_np_add_SS(3,4,5)
    boolean_ss()


