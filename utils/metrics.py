import numpy as np
from sklearn.metrics import confusion_matrix


def get_range_proba(proba, label):
    splits = np.where(label[1:] != label[:-1])[0] + 1

    is_anomaly = label[0] == 1
    new_proba = np.array(proba)
    pos = 0
    for sp in splits:
        if is_anomaly:
            new_proba[pos: sp] = np.max(proba[pos: sp])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        new_proba[pos: sp] = np.max(proba[pos: sp])
    return new_proba


def different_index(score, true):
    if len(score) != len(true):
        print('[Error] score and true have no same length in function')
        return -1
    index = []
    length = len(score)
    for i in range(length):
        if score[i] != true[i]:
            index.append(i)
    return index


def precision_recall_score(pred, label):
    '''
    Precision Recall Score
    ---
    Parameters:
        pred: detector 预测输出
        label: 数据标注
    Return
        precision
        recall
    '''
    TP = np.sum(np.logical_and(np.equal(label, 1), np.equal(pred, 1)))
    FP = np.sum(np.logical_and(np.equal(label, 0), np.equal(pred, 1)))
    TN = np.sum(np.logical_and(np.equal(label, 1), np.equal(pred, 0)))
    FN = np.sum(np.logical_and(np.equal(label, 0), np.equal(pred, 0)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return (precision, recall)


def f_score(pre, rec):
    return (2 * pre * rec) / (pre + rec)


def search_best_score(score: np.array, label: np.array, divide_num: int = 1000):
    '''
    best_f1_score
    ---
    ### Parameters:
        score: detector 经过归一化处理后的输出得分
        label: 数据标注
        divide_num: (0, 1)区间划分格子数
    ### Return:
        precision
        recall
        fscore
        threshold
    '''
    from sklearn.metrics import precision_score, recall_score, f1_score
    threshold = 0.0
    pre = []
    rec = []
    fscore = []
    for i in range(divide_num):
        threshold = threshold + 1.0/divide_num
        if threshold > 1:
            threshold = 0.99
        proba = np.floor(score - threshold + 1)
        proba = np.array(proba, dtype=int)
        proba = get_range_proba(proba, label)
        p = precision_score(label, proba, average='binary')
        r = recall_score(label, proba, average='binary')
        f = f1_score(label, proba, average='binary')
        pre.append(p)
        rec.append(r)
        fscore.append(f)
    pre = np.array(pre)
    rec = np.array(rec)
    return pre[fscore.index(max(fscore))], rec[fscore.index(max(fscore))], max(fscore), fscore.index(max(fscore))/divide_num


def dynamic_threshold(score: np.ndarray, ratio: float = 3):
    '''
    动态阈值
    ---
    ### Parameters:
        score: np.ndarray 异常评分
        ratio: float (default = 3) 标准差比例，例：threshold = mean + `ratio` * std
    ### Returns:
        proba: 预测异常
        (Temporary Delete)// threshold: 对于该段输入的score 选择的阈值
    '''
    std = np.std(score)
    mean = np.mean(score)
    threshold = mean + ratio * std
    if threshold >= 1:
        threshold = 0.999
    elif threshold <= 0:
        threshold = 0.001
    proba = np.array(score>threshold, dtype=np.int)
    proba = np.array(proba, dtype=int)
    return proba


def sliding_anomaly_predict(score: np.ndarray, window_size: int = 1440, stride: int = 10, ratio: float = 3):
    '''
    滑动窗口的动态阈值异常预测
    ---
    ### Parameters:
        score: np.ndarray 异常评分
        window_size: int (default = 1440) 滑动窗口大小
        stride: int (default = 10) 滑动窗口步长
        ratio: float (default = 3) 标准差比例，例：threshold = mean + `ratio` * std
    ### Return:
        predict: np.ndarray 根据滑动窗口的动态阈值对异常做出 0、1预测
    '''
    predict = np.zeros(score.shape, dtype=int)
    ny = score.shape[0]
    start = 0
    while start < ny:
        predict[start:start +
                window_size] = dynamic_threshold(score[start:start+window_size], ratio)
        start += stride
    return predict


def evaluate_result(proba: np.array, label: np.array):
    '''
    评价程序异常预测结果
    ---
    ### Parameters:
        proba: 程序预测输出
        label：数据标注
    ### Returns：
        p: precision 精准度
        r: recall 召回率
        f: f1 score 
    '''
    from sklearn.metrics import precision_score, recall_score, f1_score
    proba = get_range_proba(proba, label)
    p = precision_score(label, proba, average='binary')
    r = recall_score(label, proba, average='binary')
    f = f1_score(label, proba, average='binary')
    return p, r, f


def evaluation(true_label, proba, threshold=0):
    from sklearn.metrics import precision_score, recall_score, f1_score

    if not isinstance(true_label, np.ndarray):
        true_label = np.array(true_label)
    if not isinstance(proba, np.ndarray):
        proba = np.array(proba)
    new_proba = get_range_proba(proba, true_label)
    pre = []
    rec = []
    fsc = []
    if threshold:
        predict_label = np.asarray(new_proba >= threshold, dtype=np.int32)
        precision = precision_score(true_label, predict_label) * 100
        recall = recall_score(true_label, predict_label) * 100
        if precision+recall == 0:
            f_score = 0
        else:
            f_score = 2*precision*recall/(precision+recall)
        tn, fp, fn, tp = confusion_matrix(
            true_label, predict_label, labels=[0, 1]).ravel()
        return precision, recall, f_score, threshold
    for item in range(0, 100, 1):
        thr = item/100
        predict_label = np.asarray(new_proba >= thr, dtype=np.int32)
        precision = precision_score(true_label, predict_label) * 100
        recall = recall_score(true_label, predict_label) * 100
        if precision+recall == 0:
            f_score = 0
        else:
            f_score = 2*precision*recall/(precision+recall)
        pre.append(precision)
        rec.append(recall)
        fsc.append(f_score)

    best_score = max(fsc)
    best_pre = pre[fsc.index(best_score)]
    best_rec = rec[fsc.index(best_score)]
    best_thr = fsc.index(best_score)/100

    predict_label = np.asarray(new_proba >= best_thr, dtype=np.int32)
    return best_pre, best_rec, best_score, best_thr


def dynamic_best_fscore(true_label, score, window_size=1440, stride=10):
    predict = np.zeros(score.shape, dtype=int)
    ny = score.shape[0]
    start = 0
    while start < ny:
        _, _, _, threshold = evaluation(true_label[start:start+window_size], score[start:start+window_size])
        proba = np.asarray(score[start:start+window_size] > threshold, dtype=np.int)
        predict[start:start + window_size] = proba
        start += stride
    return predict

def spot_eval(init_score, score, q=1e-3, level=0.02, depth=None):
    '''
    使用SPOT方法计算阈值
    Parameter
    ---
        init_score: 启动数据，至少1000个数据点
        score: 程序测试集异常得分
        q (float): risk parameter
        level (float): Probability associated with the initial threshold t
    
    Returns
    ---
    numpy.narray
        程序对异常的预测标注
    '''
    from algorithm.spot import SPOT
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, verbose=False)  # initialization step
    results = s.run()  # run
    proba = np.array(score>results['thresholds'], dtype=np.int)

    return proba

def dspot_eval(init_score, score, q=1e-3, level=0.02, depth=1440):
    '''
    使用dSPOT方法计算阈值
    Parameter
    ---
        init_score: 启动数据，至少1000个数据点
        score: 程序测试集异常得分
        q (float, default: 1e-3): risk parameter
        level (float, default: 0.02): Probability associated with the initial threshold t
        depth: (int, default: 1440) Number of observations to compute the moving average

    Returns
    ---
    numpy.narray
        程序对异常的预测标注
    '''

    from algorithm.spot import dSPOT
    d = dSPOT(q, depth)
    d.fit(init_score, score)
    d.initialize(verbose=False)
    results = d.run()
    proba = np.array(score>results['thresholds'], dtype=np.int)

    return proba