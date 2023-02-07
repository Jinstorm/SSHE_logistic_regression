import sys
import os
import numpy as np
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_pardir = os.path.join(dir_path, os.pardir)
# abs_parpardir = os.path.join(abs_pardir, os.pardir)
sys.path.append(abs_pardir)
from paillierm.encrypt import PaillierEncrypt
# from paillierm.fixedpoint import FixedPointEndec

en = PaillierEncrypt()
en.generate_key(2048)

# a = np.random.random((1000))
# a2 = a.tolist()
# a = a * 20000
# s = time.time()
# en_rs = en.recursive_encrypt(a)
# e = time.time()
# print(e-s)

tmp = en.recursive_encrypt(1)
tmpl = en.recursive_decrypt(tmp)
print(tmpl)
