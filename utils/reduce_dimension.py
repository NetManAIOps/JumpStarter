import math
import numpy as np


def reduce_dimension(x, Orig, method='Mean'):
    '''
    ## Reduce the dimension of the metrix
    ### Parameters
        x: the input ndarray.
        method: choose the method used to reduce dimension of x.
    ### Method:\n
        1. Mean: Get the abs value of a row, and use the mean of the row to replace the row.
        2. Euclidean: 
        3. Manhattan:
        4. Chebyshev:
        5. Cosine:
    '''
    ny, nx = x.shape
    Xrd = abs(x)
    if method == 'Mean':
        Xrd = abs(Orig-x)
        Xrd = Xrd.sum(axis=1)

    elif method == 'Euclidean':
        Xrd = [np.linalg.norm(Orig[t]-x[t]) for t in range(ny)]

    elif method == 'Manhattan':
        Xrd = [np.linalg.norm(Orig[t]-x[t], ord=1) for t in range(ny)]

    elif method == 'Chebyshev':
        Xrd = [np.linalg.norm(Orig[t]-x[t], ord=np.inf) for t in range(ny)]

    elif method == 'Cosine':
        Xrd = [np.dot(Orig[t], x[t])/(np.linalg.norm(Orig[t]) *
                                      (np.linalg.norm(x[t]))) for t in range(ny)]

    return Xrd

# from sklearn.preprocessing import normalize


def norm(x, method='linear'):
    '''
    ## Normalized
    ### Parameters
        x: the input vector
        method: ...
    ### Methods:
        1. linear
        2. z-score
        3. atan
        4. sigmod
        5. tanh
    '''
    if method == 'linear':
        maxVal = max(x)
        minVal = min(x)
        dis = maxVal - minVal
        x = (x-minVal)/dis

    elif method == 'z-score':
        x = (x-np.mean(x))/np.std(x)

    elif method == 'atan':
        y = [math.atan(t)*2/math.pi for t in x]
        x = y

    elif method == 'sigmod':
        x = [1 / (1 + np.exp(-t)) for t in x]

    elif method == 'tanh':
        s1 = [np.exp(t) - np.exp(-t) for t in x]
        s2 = [np.exp(t) + np.exp(-t) for t in x]
        x = [s1[t] / s2[t] for t in range(len(s1))]

    return x
