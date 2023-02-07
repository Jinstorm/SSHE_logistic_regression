import numpy as np
import matplotlib.pyplot as plt

dataset_name = "ionosphere.data"

lr = .1
batch_size = 16
n_iterations = 10000

patience = 2
min_delta = 1e-2

sigmoid = lambda z : 1 / (1 + np.exp(-z))
logloss = lambda y_hat, y : np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) / len(y_hat)

def gradient_descent(X, y, beta, lr):
    y = y.reshape(-1, 1)
    gradients = np.dot(X.T, sigmoid(np.dot(X, beta.T)) - y) / len(y)
    new_betas = beta - lr * gradients.T

    return new_betas

def prepare_batches(X, y, batch_size):
    X_batch_list = list()
    y_batch_list = list()
    
    for i in range(len(y) // batch_size):
        X_batch_list.append(X[i * batch_size : i * batch_size + batch_size, :])
        y_batch_list.append(y[i * batch_size : i * batch_size + batch_size])
    
    if len(y) % batch_size > 0:
        X_batch_list.append(X[len(y) // batch_size * batch_size:, :])
        y_batch_list.append(y[len(y) // batch_size * batch_size:])

    return X_batch_list, y_batch_list


predict = lambda X: sigmoid(np.dot(X, betas.T)) > .5
predict_with_output = lambda X: (X > .5) * 1

f = open(dataset_name, "r")

X = list()
y = list()

encode_char = ""
encoder = lambda y : 0 if y is encode_char else 1

for row in f:
    split_row = [x.strip() for x in row.split(',')]
    if encode_char is "":
        encode_char = split_row[-1]
    y.append(encoder(split_row[-1]))
    X.append([np.array(split_row[:-1]).astype(np.float)])





permutations = np.random.permutation(len(X))

X, y = np.asarray(X).squeeze(), np.asarray(y)

X = X[permutations, :]
y = y[permutations]

#To add beta 0
temp = np.ones((X.shape[0], X.shape[1] + 1))
temp[:, 1:] = X
X = temp

len_test = len(X) // 5 
len_train = len(X) - len_test
X_test, y_test, X_train, y_train = X[:len_test, :], y[:len_test], X[len_test:, :], y[len_test:]
betas = np.random.random(X.shape[1]).reshape(1, -1)




print("Train..")
train_error_hist = list()
test_error_hist = list()
test_acc_hist = list()

X_batch_list, y_batch_list = prepare_batches(X_train, y_train, batch_size)

n_batches = len(y_batch_list)

prev_average = 10000

patience_counter = 0
iteration_counter = 0 
while iteration_counter < n_iterations:
    for i in range(n_batches):
        X_batch = X_batch_list[i]
        y_batch = y_batch_list[i]

        betas = gradient_descent(X_batch, y_batch, betas, lr)
        
        y_hat = sigmoid(np.dot(X_batch, betas.T))
        train_error_hist.append(logloss(y_hat, y_batch) / len(y_batch))
              
        y_hat = sigmoid(np.dot(X_test, betas.T))
        test_error_hist.append(logloss(y_hat, y_test) / len(y_test))
        test_acc_hist.append(np.mean((predict_with_output(y_hat) == y_test.reshape(-1, 1)) * 1))
          
        iteration_counter += 1
        
    current_average = np.mean(train_error_hist[-n_batches:])
        
    if np.abs(prev_average - current_average) < min_delta:
        patience_counter += 1
    else:
        patience_counter = 0
        
    prev_average = current_average
    
    if patience_counter == patience:
        break


plt.plot(test_error_hist)
plt.plot(train_error_hist)
plt.xlabel("#Iterations")
plt.ylabel("Total Loss")
plt.title("Loss vs Number of iterations")
plt.legend(("Test error", "Train error"))

plt.plot(test_acc_hist)
plt.xlabel("#Iterations")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of iterations")