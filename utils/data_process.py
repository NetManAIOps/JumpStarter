import numpy as np


def median(num_list: list):
    """
    求列表的中位数
    :param 数列表
    :return:
    """
    num_list.sort()
    half = len(num_list) // 2
    return (num_list[half] + num_list[~half]) / 2


def normalization(data):
    _range = np.max(data) - np.min(data)
    if _range == 0:
        return np.zeros_like(data) + 0.5
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def smooth(a, n, mode="same"):
    return np.convolve(a, np.ones((n,)) / n, mode=mode)
