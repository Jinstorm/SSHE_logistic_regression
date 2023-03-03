import numpy as np

# secret_number_array = np.random.randint(0, 2, (0, 1000))  # 1行m列的y向量（y属于0或1）
# b=np.random.randint(0,2,size=(5,4))
# # a = np.array([2,3,4,5,6])
# a = np.random.randint(0, 2, (4,1)) # 4000行1列的列向量，如果要生成数组使用(4000,)  # np.arange(4000)
# a = a.reshape(-1) # lie向量转化为数组
# print(a)
# print(b)
# a = np.dot(b,a).tolist()
def transpose_np():
    np.random.seed(10)
    arr = np.random.randint(0,10,size=(5,4))
    print("origin arr: \n", arr)
    trans = arr.transpose()
    print("transpose arr: \n", trans)
    ''' ^output^
    origin arr: 
    [[9 4 0 1]
    [9 0 1 8]
    [9 0 8 6]
    [4 3 0 4]
    [6 8 1 8]]
    transpose arr: 
    [[9 9 9 4 6]
    [4 0 0 3 8]
    [0 1 8 0 1]
    [1 8 6 4 8]]
    '''
def matrix_shape():
    arr1 = np.random.randint(0,10,size=(5,4))
    arr2 = np.random.randint(0,10,size=(5,4))

    assert(arr1.shape == arr2.shape)
    print("Bingo.")

def array_length():
    batch_num = []
    for i in range(10):
        batch_num.append(i)
    print(len(batch_num))

def numpy_row_col():
    np.random.seed(10)
    arr1 = np.random.randint(0,10,size=(5,4))
    print(arr1)
    print(arr1[2], arr1[2].shape)

    print(arr1[0:2,0], arr1[0:2,0].shape)

def test_Mul():
    mat1 = np.random.randint(0,10,size=(5,4))
    mat2 = np.random.randint(0,10,size=(4,5))

    res = np.dot(mat1, mat2[:,0])
    print(res)

if __name__ == "__main__":
    # transpose_np()
    # matrix_shape()
    # array_length()
    # numpy_row_col()
    test_Mul()
    pass