import numpy as np

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        val = x[idx]
        x[idx] = val + h
        f1 = f(x)

        x[idx] = val - h
        f2 = f(x)

        grad[idx] = (f1 - f2) / (2*h)
        x[idx] = val

    return grad
