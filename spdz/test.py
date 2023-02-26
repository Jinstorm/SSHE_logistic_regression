import multiprocessing
import time

def worker(d, key, value):
    d[key] = value
 
if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    d = mgr.dict()
    jobs = [ multiprocessing.Process(target=worker, args=(d, i, i*2))
             for i in range(10)
             ]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    print ('Results:' )
    for key, value in enumerate(dict(d)):
        print("%s=%s" % (key, value))
         
# the output is :
# Results:
# 0=0
# 1=1
# 2=2
# 3=3
# 4=4
# 5=5
# 6=6
# 7=7
# 8=8
# 9=9
