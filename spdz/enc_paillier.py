from phe import paillier
import numpy as np
import pandas as pd
# generate a public key and private key pair
public_key, private_key = paillier.generate_paillier_keypair()

# start encrypting numbers
# secret_number_list = [3.141592653, 300, -4.6e-12]
secret_number_array = np.random.randint(0, 2, (0, 1000))  # 1行m列的y向量（y属于0或1）
b=np.random.randint(0,2,size=(1000,4000))
# a = np.array([2,3,4,5,6])
a = np.random.randint(0, 2, (4000,1)) # np.arange(4000)
a = a.reshape(-1)
# b = b = np.arange(4).reshape((2,5))
print(a)
print(b)
a = np.dot(b,a).tolist()
print(type(a), a[0], a[1])
print(a)
# print(np.dot(a,b))

# print(secret_number_array)
# a = np.arange(0,1000,1)
# a = secret_number_array
# a = a.tolist() #reshape(-1)
# a = pd.
# print(a.type)
# print(a.dtype)
# a = pd.Series(a)
# a.apply(public_key.encrypt)
encrypted_number_array = np.asarray([public_key.encrypt(x) for x in a]) # 矩阵1000*4000 向量4000*1 = 30s
# encrypted_number_array = np.apply_along_axis(public_key.encrypt, 0, a) # 可能是对每一维度操作，所以识别不到每一个元素
# encrypted_number_list = [public_key.encrypt(x) for x in a]
# print(a[0], a[1], a[2])


# decrypt encrypted number and print
# print([private_key.decrypt(x) for x in encrypted_number_list])
# print(np.apply_along_axis(private_key.decrypt, 0, encrypted_number_array))