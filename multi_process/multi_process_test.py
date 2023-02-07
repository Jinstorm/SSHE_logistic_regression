from multiprocessing import Pool
import time
# from federatedml.secureprotol import PaillierEncrypt
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
abs_pardir = os.path.join(dir_path, os.pardir)
sys.path.append(abs_pardir)
from paillierm.encrypt import PaillierEncrypt
import numpy as np

np.random.seed(100)
a = np.random.random(5000)
cipher = PaillierEncrypt()
cipher.generate_key(2048)

def encrypt_vector(tmp):
    encrypt_table = cipher.recursive_encrypt(tmp)
    return encrypt_table

if __name__ == '__main__':
    pool = Pool(4)
    # args = [a,]
    start = time.time()
    print("hello")
    ret = pool.map(cipher.recursive_encrypt, a)  # 不会阻塞
    print("hello")
    pool.close()
    pool.join()
    # print(ret.get()[1])
    # print(ret)
    # print(cipher.recursive_decrypt(ret))
    print(f"It takes {time.time() - start} seconds!")

    start = time.time()
    encrypt_table = cipher.recursive_encrypt(a)
    print(f"It takes {time.time() - start} seconds!")