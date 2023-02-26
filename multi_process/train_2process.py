import multiprocessing
from multiprocessing import Process, Queue, set_start_method
import time,random,os
import sys
import numpy as np
import time

# dir_path = os.path.dirname(os.path.realpath(__file__))
# abs_pardir = os.path.join(dir_path, os.pardir)
# # abs_parpardir = os.path.join(abs_pardir, os.pardir)
# sys.path.append(abs_pardir)
# from paillierm.encrypt import PaillierEncrypt
# # from paillierm.fixedpoint import FixedPointEndec

# en = PaillierEncrypt()
# en.generate_key(2048)

# # a = np.random.random((1000))
# # a2 = a.tolist()
# # a = a * 20000
# # s = time.time()
# # en_rs = en.recursive_encrypt(a)

# tmp = en.recursive_encrypt(1)
res1 = np.random.random((1000))
res2 = np.random.random((1000))
res3 = np.random.random((1000))
res4 = np.random.random((1000))
res5 = np.random.random((1000))
res = [res1,res2,res3,res4,res5]
class test:
    def __init__(self):
        self.num = 100
    # def consumer(self, q, p):
    #     while True:
    #         # time.sleep(random.randint(1,3))
    #         res='子%s'
    #         q.put(res)
    #         res=p.get()
    #         if res is None: break #收到结束信号则结束
    #         # time.sleep(random.randint(1,3))
    #         # print('\033[45m%s 吃 %s\033[0m' %(os.getpid(),res))
    #         print('\033[45m%s 吃 %s\033[0m' %(os.getpid(), res))


    # def producer(self, q, p):
    #     for i in range(5):
    #         time.sleep(random.randint(1,3))
    #         res='包子%s' %i
    #         p.put(res)
    #         res=q.get()
    #         print('\033[46m%s 生产了 %s\033[0m' %(os.getpid(),res))
    #     q.put(None) #发送结束信号
    
    def producer(self, q, p):
        
        # time.sleep(random.randint(1,2))
        res1=1
        res2=2
        p.put(res1)
        
        # time.sleep(random.randint(1,2))
        p.put(res2)
        # res=q.get()
        # print('\033[46m%s 生产了 %s\033[0m' %(os.getpid(),res))
        # time.sleep(random.randint(1,3))
        q.put(None) #发送结束信号
        p.put(res2)
        print("ahahahaahh")
    
    def consumer(self, q, p):
       
        time.sleep(6)
        # res='子%s'
        # q.put(res)
        res1=p.get()
        
        res2=p.get()
        print("res1: ", res1)
        print("res2: ", res2)
        if res is None: print("hello") #收到结束信号则结束
        res3=p.get()
        print("res3: ", res3)
        self.num = 2
        print("num: ", self.num)
        p.put(self.num)
        # time.sleep(random.randint(1,3))
        # print('\033[45m%s 吃 %s\033[0m' %(os.getpid(),res))
        # print('\033[45m%s 吃 %s\033[0m' %(os.getpid(), res))


if __name__ == '__main__':
    methods = multiprocessing.get_all_start_methods()
    print(methods)
    # set_start_method('fork')
    obj = test()
    q=Queue()
    p=Queue()
    #生产者们:即厨师们
    p1=Process(target=obj.producer,args=(q, p,))
    #消费者们:即吃货们
    c1=Process(target=obj.consumer,args=(q, p,))

    #开始
    p1.start()
    c1.start()
    print('进程间通信-队列-主进程')
    p1.join()
    c1.join()
    print("num: ", obj.num)
    obj.num = p.get()
    print("num: ", obj.num)