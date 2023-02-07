import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

# _row  = np.array([0, 3, 1, 0])
# _col  = np.array([0, 3, 1, 2])
# _data = np.array([4, 5, 7, 9])
# coo = coo_matrix((_data, (_row, _col)), shape=(4, 4), dtype=np.int)
# coo.todense()  # 通过toarray方法转化成密集矩阵(numpy.matrix)
# coo.toarray()  # 通过toarray方法转化成密集矩阵(numpy.ndarray)
# # array([[4, 0, 9, 0],
# #        [0, 7, 0, 0],
# #        [0, 0, 0, 0],
# #        [0, 0, 0, 5]])

from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand
A = lil_matrix((1000, 1000))
A[0, :100] = rand(100)
A[1, 100:200] = A[0, :100]
A.setdiag(rand(1000))
# 现在将其转换为 CSR 格式并求解 A x = b for x：

A = A.tocsr()
b = rand(1000)
# x = spsolve(A, b)
# 将其转换为稠密矩阵并求解，并检查结果是否相同：

# x_ = solve(A.toarray(), b)
# 现在我们可以用以下方法计算误差范数：
z = np.dot(A.toarray(), b)
z_ = A.dot(b)
print(type(z_))
print(z_ * 0.25 + 0.5)
err = norm(z-z_)
print(err)
# err < 1e-10
# True
