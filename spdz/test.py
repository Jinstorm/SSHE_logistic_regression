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
a = np.random.randint(0, 2, (4000,1)) # 
a = a.reshape(-1)
b = np.arange(4000)
print(type(a), type(b))
print(a.shape, b.shape)