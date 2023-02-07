import numpy as np

secret_number_array = np.random.randint(0, 2, (0, 1000))  # 1行m列的y向量（y属于0或1）
b=np.random.randint(0,2,size=(5,4))
# a = np.array([2,3,4,5,6])
a = np.random.randint(0, 2, (4,1)) # 4000行1列的列向量，如果要生成数组使用(4000,)  # np.arange(4000)
a = a.reshape(-1) # lie向量转化为数组
print(a)
print(b)
a = np.dot(b,a).tolist()