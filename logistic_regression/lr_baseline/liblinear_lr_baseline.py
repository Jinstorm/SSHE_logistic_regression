import numpy as np
import scipy
from liblinear.liblinearutil import *


def read_sampling_data():
    from sklearn.datasets import load_svmlight_file
    import os

    # dataset_file_name = 'splice'
    dataset_file_name = 'a6a'
    # train_file_name = 'splice_train.txt'
    train_file_name = 'a6a.txt'
    # test_file_name = 'splice_test'
    test_file_name = 'a6a.t'
    # main_path = '/Users/zbz/code/vscodemac_python/hetero_sshe_logistic_regression/data/'
    main_path = '/Users/zbz/data/'
    train_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, train_file_name))
    test_data = load_svmlight_file(os.path.join(main_path, dataset_file_name, test_file_name))
    X_train = train_data[0]
    Y_train = train_data[1].astype(int)
    X_test = test_data[0]
    Y_test = test_data[1].astype(int)
    # print(X_train) # 1000 * 60
    # print(Y_train[0]) # 1000 * 1
    return X_train.todense().A, Y_train, X_test.todense().A, Y_test # matrix转array


X_train, y_train, X_test, y_test = read_sampling_data()

model = train(y_train, X_train, '-s 0 -c 1')
p_label, p_acc, p_val = predict(y_test, X_test, model)


