
# coding: utf-8

# In[3]:


import numpy as np

def step_func(val):
    if val > 0:
        return 1
    else:
        return 0
    
X = np.array([2,3])
W = np.random.randn(2,1)
b = 1 

result = np.dot(X,W) + 1


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val # 값 복원
        it.iternext()   

