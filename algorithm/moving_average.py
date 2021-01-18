import numpy as np


def moving_average(x: np.array, window: int, stride: int):
    """
    滑动窗口平均
    :param x: 数据矩阵 shape=(n,d), n是采样点个数, d是每个点数据维度
    :param window: 窗口长度
    :param stride: 窗口步长
    :return:
    """
    n, k = x.shape
    if window > n:
        window = n
    score = np.zeros(n)
    # 记录每个点被更新的次数, 在新的点的影响加入进来时, 表示该点原来值的权重
    score_weight = np.zeros(n)
    # 窗口起始
    wb = 0
    while True:
        # 窗口结束
        we = wb + window
        x_window = x[wb:we]
        # 该窗口的平均向量
        x_mean = np.mean(x_window, axis=0)
        # 给score加上影响
        dis = np.sqrt(np.sum(np.square(x_window - x_mean), axis=1))
        score[wb:we] = (score[wb:we] * score_weight[wb:we] + dis)
        score_weight[wb:we] += 1
        score[wb:we] /= score_weight[wb:we]
        if we >= n:
            break
        wb += stride
    # 将score映射到(0, 1]上
    return score


def online_moving_average(
        incoming_data: np.array, historical_data: np.array,
        window: int, stride: int
):
    """
    在线滑动窗口平均
    :param incoming_data: 当前数据矩阵 shape=(n,d), n是需要计算的点个数, d是每个点数据维度
    :param historical_data: 历史数据矩阵 shape=(m,d), m是历史时间点个数, d是每个点数据维度,
    :param window: 窗口长度
    :param stride: 窗口步长
    :return:
    """
    n = incoming_data.shape[0]
    # 根据窗口大小计算出所需要的所有数据量(把第一个窗口的终点放在incoming_data的起点)
    need_history_begin = max(0, historical_data.shape[0] - window + 1)
    need_data = np.concatenate(
        [incoming_data, historical_data[need_history_begin:]], axis=0
    )
    score = moving_average(need_data, window, stride)
    # 截取最后n个点的数据表示incoming_data的数值将score映射到(0, 1]上
    return score[0:n]
