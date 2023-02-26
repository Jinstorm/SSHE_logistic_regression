import numpy as np
# r=np.random.random
a = list(np.array([1,2,3,4,5,6]))
b = list(np.array([11,12,13,14,15,16]))
print(a,b)
np.random.seed(2)
np.random.shuffle(a)
np.random.seed(2)
np.random.shuffle(b)
print(a,b)


