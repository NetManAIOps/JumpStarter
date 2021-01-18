# from kshape.core import kshape
import pandas as pd
import numpy as np
import math
import numpy as np

from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft
from sklearn import datasets
from sklearn.cluster import DBSCAN


def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)


def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate(
            (a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def _ncc_c(x, y):
    """
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
    """
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    return np.real(cc) / den


def _ncc_c_2dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 1 dimensional
    y vector
    Returns a 2 dimensional array of normalized fourier transforms
    """
    den = np.array(norm(x, axis=1) * norm(y))
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[:, -(x_len-1):], cc[:, :x_len]), axis=1)
    return np.real(cc) / den[:, np.newaxis]


def _ncc_c_3dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 2 dimensional
    y vector
    Returns a 3 dimensional array of normalized fourier transforms
    """
    den = norm(x, axis=1)[:, None] * norm(y, axis=1)
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = np.concatenate((cc[:, :, -(x_len-1):], cc[:, :, :x_len]), axis=2)
    return np.real(cc) / den.T[:, :, None]


def _sbd(x, y):
    """
    >>> _sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift


def _extract_shape(idx, x, j, cur_center):
    """
    >>> _extract_shape(np.array([0,1,2]), np.array([[1,2,3], [4,5,6]]), 1, np.array([0,3,4]))
    array([-1.,  0.,  1.])
    >>> _extract_shape(np.array([0,1,2]), np.array([[-1,2,3], [4,-5,6]]), 1, np.array([0,3,4]))
    array([-0.96836405,  1.02888681, -0.06052275])
    >>> _extract_shape(np.array([1,0,1,0]), np.array([[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]]), 0, np.array([0,0,0,0]))
    array([-1.2089303 , -0.19618238,  0.19618238,  1.2089303 ])
    >>> _extract_shape(np.array([0,0,1,0]), np.array([[1,2,3,4],[0,1,2,3],[-1,1,-1,1],[1,2,2,3]]), 0, np.array([-1.2089303,-0.19618238,0.19618238,1.2089303]))
    array([-1.19623139, -0.26273649,  0.26273649,  1.19623139])
    """
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            if cur_center.sum() == 0:
                opt_x = x[i]
            else:
                _, opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)
    a = np.array(_a)

    if len(a) == 0:
        return np.zeros((1, x.shape[1]))
    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)
    s = np.dot(y.transpose(), y)

    p = np.empty((columns, columns))
    p.fill(1.0/columns)
    p = np.eye(columns) - p

    m = np.dot(np.dot(p, s), p)
    _, vec = eigh(m)
    centroid = vec[:, -1]
    finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
    finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())

    if finddistance1 >= finddistance2:
        centroid *= -1

    return zscore(centroid, ddof=1)


def _kshape(x, k):
    """
    >>> from numpy.random import seed; seed(0)
    >>> _kshape(np.array([[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]]), 2)
    (array([0, 0, 1, 0]), array([[-1.2244258 , -0.35015476,  0.52411628,  1.05046429],
           [-0.8660254 ,  0.8660254 , -0.8660254 ,  0.8660254 ]]))
    """
    m = x.shape[0]  # 一共有m组数据
    idx = randint(0, k, size=m)  # 生成k个最小为0的数据
    centroids = np.zeros((k, x.shape[1]))  # k行，timestamps列
    distances = np.empty((m, k))  # m行 k列

    for _ in range(100):
        old_idx = idx
        for j in range(k):
            centroids[j] = _extract_shape(idx, x, j, centroids[j])

        distances = (1 - _ncc_c_3dim(x, centroids).max(axis=2)).T

        idx = distances.argmin(1)
        if np.array_equal(old_idx, idx):
            break

    return idx, centroids


def kshape(x, k):
    idx, centroids = _kshape(np.array(x), k)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))
    return clusters


#================辅助函数==================#
def get_idx(idx):
    ''' 获得下标 '''
    x = idx[0][0]
    y = idx[1][0]
    if y > x:
        temp = y
        y = x
        x = temp
    return x, y
#=================一些约定=================#
# 叶子节点是这样的. --> ['leaf', label]
# 中心节点是这样的. --> ['cluster', distance, [left tree], [right tree]] distance代表左子树和右子树的距离


def get_type(node):
    ''' 获取叶子节点的类型 '''
    return node[0]


def get_label(node):
    ''' 获取叶子节点的label '''
    assert(get_type(node) == 'leaf')
    return node[1]


def get_left(node):
    ''' 获取左子树 '''
    assert(get_type(node) == 'cluster')
    return node[2]


def get_right(node):
    ''' 获取右子树 '''
    assert(get_type(node) == 'cluster')
    return node[3]


def get_distance(node):
    ''' 获取距离值 '''
    assert(get_type(node) == 'cluster')
    return node[1]


def make_leaf(label):
    ''' 构建叶子节点 '''
    return ['leaf', label]


def make_cluster(distance, left, right):
    ''' 构建中心节点 '''
    return ['cluster', distance, left, right]

#=================直接聚类=================#


def get_leaf_labels(node):
    ''' 获取标签的值 '''
    # 从node节点出发,遍历叶子节点
    node_type = get_type(node)
    if node_type == 'leaf':
        return [get_label(node)]
    elif node_type == 'cluster':
        labels = get_leaf_labels(get_left(node))
        labels.extend(get_leaf_labels(get_right(node)))
        return labels


def get_classify(distance, node):
    ''' 获得分类的数据 '''
    node_type = get_type(node)
    if node_type == 'leaf':
        return [[get_label(node)]]
    elif node_type == 'cluster' and get_distance(node) < distance:  # 要开始分类了
        return [get_leaf_labels(node)]
    else:
        llabels = get_classify(distance, get_left(node))
        rlabels = get_classify(distance, get_right(node))
        llabels.extend(rlabels)
        return llabels


def direct_cluster(simi_matrix):
    ''' 直接聚类法 '''
    # 为了尽量简单,我只用list来构建树
    N = len(simi_matrix)  # 得到矩阵对应的点的数目
    nodes = [make_leaf(label) for label in range(N)]  # 构建一堆叶子节点
    np.fill_diagonal(simi_matrix, float('Inf'))  # 先用最大值填充对角线
    root = 0  # 用于记录根节点的下标
    while N > 1:
        # 然后视图寻找相似度矩阵中最小的那个数的下标
        idx = np.where(simi_matrix == simi_matrix.min())
        x, y = get_idx(idx)  # 得到下标值==> x 和 y 进行了合并
        distance = simi_matrix[x][y]  # 得到两者的距离
        cluster = make_cluster(distance, nodes[x], nodes[y])
        nodes[y] = cluster  # 更新
        root = y  # 更新下标值
        # 删除x行, y列的元素
        simi_matrix[x] = float('Inf')
        simi_matrix[:, x] = float('Inf')
        N = N - 1
    return nodes[root]


def cluster(X: np.ndarray, threshold: float = 0.01):
    '''
    BSD - 聚类
    ---
    Parameters：
        X: numpy.ndarray shape = (m, n)的数组
        threshold: float (default: 0.01) 直接聚类的阈值
    Return
        元素是numpy.ndarray的 numpy.ndarray, 其中的元素为 0 ~ (n-1)的整型数字，在同一个列表中即为一类
    '''
    ny, nx = X.shape

    distance = np.zeros((nx, nx))
    for (i, j), v in np.ndenumerate(distance):
        distance[i, j] = _sbd(X[:, i], X[:, j])[0]
    tree = direct_cluster(distance)
    return (get_classify(threshold, tree))


if __name__ == "__main__":
    import sys
    import doctest
    sys.exit(doctest.testmod()[0])
