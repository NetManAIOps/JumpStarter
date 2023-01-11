import scipy.fftpack as spfft
import cvxpy as cvx
import numpy as np


def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho',
                     axis=0)


def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho',
                      axis=0)


def reconstruct(n, d, index, value):
    """
    压缩感知采样重建算法
    :param n: 需重建数据的数据量
    :param d: 需重建数据的维度
    :param index: 采样点的时间维度坐标 属于[0, n-1]
    :param value: 采样点的KPI值，shape=(m, d), m为采样数据量
    :return:x_re: 重建的KPI数据，shape=(n, d)
    """
    x = np.zeros((n, d))
    x[index, :] = value

    a = index
    b = index

    if d > 1:
        for i in range(d - 1):
            a = a + n * 1
            b = np.concatenate((b, a))
    index = b.astype(int)

    # random sample of indices

    ri = index
    b = x.T.flat[ri]
    # b = value.T.flat
    # b = np.expand_dims(b, axis=1)

    # create dct matrix operator using kron (memory errors for large ny*nx)
    transform_mat = np.kron(
        spfft.idct(np.identity(d), norm='ortho', axis=0),
        spfft.idct(np.identity(n), norm='ortho', axis=0)
    )
    transform_mat = transform_mat[ri, :]  # same as phi times kron

    # do L1 optimization
    vx = cvx.Variable(d * n)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [transform_mat @ vx == b]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='OSQP')
    x_transformed = np.array(vx.value).squeeze()
    # reconstruct signal
    x_t = x_transformed.reshape(d, n).T  # stack columns
    x_re = idct2(x_t)

    # confirm solution
    if not np.allclose(b, x_re.T.flat[ri]):
        print('Warning: values at sample indices don\'t match original.')

    return x_re
