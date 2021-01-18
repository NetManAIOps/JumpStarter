import numpy as np
import random


def similarity(x, y):
    """
    计算两个向量的相似度
    :param x:
    :param y:
    :return:
    """
    return 1 / (1 + np.sqrt(np.sum(np.square(x - y))))


def online_lesinn(
        incoming_data: np.array,
        historical_data: np.array,
        t: int = 50,
        phi: int = 20,
        random_state: int = None
):
    """
    在线离群值算法 lesinn
    :param incoming_data: shape=(m, d,) 需要计算离群值的向量
    :param historical_data: shape=(n,d) 历史数据
    :param t: 每个数据点取t个data中子集作为离群值参考
    :param phi: 每个数据t个子集的大小
    :param random_state: 随机数种子
    :return:
    """
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)
    m = incoming_data.shape[0]
    # 将历史所有数据和需要计算离群值的数据拼接到一起
    if historical_data.shape:
        all_data = np.concatenate([historical_data, incoming_data], axis=0)
    else:
        all_data = incoming_data
    n, d = all_data.shape
    data_score = np.zeros((m,))
    sample = set()
    for i in range(m):
        score = 0
        for j in range(t):
            sample.clear()
            while len(sample) < phi:
                sample.add(random.randint(0, n - 1))
            nn_sim = 0
            for each in sample:
                nn_sim = max(
                    nn_sim, similarity(incoming_data[i, :], all_data[each, :])
                )
            score += nn_sim
        if score:
            data_score[i] = t / score
    return data_score


def lesinn(data, t=50, phi=20, random_state=None):
    """
    :param data: 数据矩阵, 行主序 shape=(n, d) 数据量n 数据维度d
    :param t: 每个数据点取t个data中子集作为离群值参考
    :param phi: 每个数据t个子集的大小
    :param random_state: 指定随机种子, 如果为None则是不指定
    :return: 每个data元素的离群值数组
    """
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)
    n = data.shape[0]
    data_score = np.zeros((n, ))
    sample = set()
    for i in range(n):
        score = 0
        for j in range(t):
            sample.clear()
            while len(sample) < phi and len(sample) < n:
                sample.add(random.randint(0, n - 1))
            nn_sim = 0
            for each in sample:
                nn_sim = max(nn_sim, similarity(data[i, :], data[each, :]))
            score += nn_sim
        if score:
            data_score[i] = (t / score)
    return data_score


def toy():
    """
    用二维高斯分布点测试lesinn算法:
    :return:
    """
    from matplotlib import pyplot as plt
    import time
    n = 1000
    data = []
    for p in range(n):
        data.append(np.random.normal(0, 1, 2))
    s = lesinn(np.array(data))
    data_set = np.mat(data)
    t1 = time.time()
    print(time.time() - t1)
    ds = []
    for p in range(n):
        ds.append(np.array([data_set[p][0], data_set[p][1], s[p]]))
    ds.sort(key=lambda x: x[2], reverse=True)
    ano = []
    for p in range(int(n / 20)):
        ano.append(ds[p])
    ano = np.array(ano)
    plt.figure()
    plt.scatter(data_set[:, 0], data_set[:, 1])
    plt.scatter(ano[:, 0], ano[:, 1], c='r', marker='x')
    plt.show()


if __name__ == '__main__':
    toy()
