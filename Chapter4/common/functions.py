import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t])) / batch_size


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_a = np.sum(exp_a)
    return exp_a / sum_a

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))