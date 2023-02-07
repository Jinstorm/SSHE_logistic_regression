import sys
import numpy as np
# if not "/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/paillier/" in sys.path:
#     sys.path.append("/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/paillier/") 
# if not 'gmpy_math' in sys.modules:  #这里a是模块名
#     gmpy_math = __import__('gmpy_math')
# else:
#     eval('import gmpy_math')
#     gmpy_math = eval('reload(gmpy_math)')

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# sys.path.insert(0, parent_dir_path)
# sys.path.append('/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression')

abs_pardir = os.path.join(dir_path, os.pardir)
abs_parpardir = os.path.join(abs_pardir, os.pardir)
sys.path.append(abs_parpardir)
print(abs_parpardir)

# import paillierm.encrypt
from paillierm.encrypt import PaillierEncrypt
from paillierm.fixedpoint import FixedPointEndec
from paillierm.utils import urand_tensor
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
# import fate_paillier
print("hello")
vector = np.array([[1,2,3,4,5]])
a = np.ones((10,2))
A = lil_matrix(a)
# init model
cipher = PaillierEncrypt()
cipher.generate_key()
fixedpoint_encoder = FixedPointEndec(n = 1e10)


##### 矩阵乘加密向量 (支持任意多次的向量数乘和加法, 成功解密)
encrypt_mat = cipher.recursive_encrypt(vector.T)
print("encrypt_mat: shape - ", encrypt_mat.shape)
print("encrypt_mat.type: ", type(encrypt_mat))
print("point type: ", type(encrypt_mat[0][0]))
print("point value: ", encrypt_mat[0][0])
matrix = np.array([[1,2,2,1,2],[1,2,2,1,2]])
# matrix = A
# mul_result = A.dot(encrypt_mat.astype(float))
mul_result = np.dot(matrix, encrypt_mat)
print(mul_result)
# mul_result = np.dot(np.array([[3]]), mul_result)
# mul_result = np.array([[3,1,1,1,1]]) + mul_result

# print(cipher.recursive_decrypt(mul_result))
# print(np.dot(matrix, vector))



# SS
base = 10
frac = 4
# encoder = FixedPointEndec(base=base, precision_fractional=frac)
_pre = urand_tensor(q_field = fixedpoint_encoder.n, tensor = mul_result)
tmp = fixedpoint_encoder.decode(_pre)  # 对每一个元素decode再返回, shape不变
share = mul_result - tmp
# share = mul_result - encoder.decode(_pre)
print("mul: ", mul_result[0][0])
print("tmp: ", tmp[0][0])
print("share: ", mul_result[0][0] - tmp[0][0])
print("share: ", (mul_result - tmp)[0][0])
print("share: ", share[0][0])

# print("encrypt_mat: ", encrypt_mat)
# print("mul_result: ", mul_result)
# print("_pre: ", _pre)
# print("share: ", share)

# print((encoder.decode(tmp) + share)[0][0])

# if mul_result[0][0] == (encoder.decode(tmp) + share)[0][0]: print("ok.")
# if 1==1: print("damn")
print(type(share))
print(type(tmp))
share += tmp
try:
    share %= fixedpoint_encoder.n
    print(share)
except BaseException:
    print("Damn.")
    print(share)
    result = cipher.recursive_decrypt(share)
    print(result)
    print(type(result))
    print(type(result[0][0]))
    
