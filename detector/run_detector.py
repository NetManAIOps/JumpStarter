import numpy as np
import pandas as pd
import yaml


def anomaly_score_example(source: np.array, reconstructed: np.array):
    """
    Calculate anomaly score
    :param source: original data
    :param reconstructed: reconstructed data
    :return:
    """
    n, d = source.shape
    d_dis = np.zeros((d,))
    for i in range(d):
        dis = np.abs(source[:, i] - reconstructed[:, i])
        dis = dis - np.mean(dis)
        d_dis[i] = np.percentile(dis, anomaly_score_example_percentage)
    if d <= anomaly_distance_topn:
        return d / np.sum(1 / d_dis)
    topn = 1 / d_dis[np.argsort(d_dis)][-1 * anomaly_distance_topn:]
    return anomaly_distance_topn / np.sum(topn)


def p_normalize(x: np.array):
    """
    Normalization
    :param: data
    :return:
    """
    p_min = 0.05
    x_max, x_min = np.max(x), np.min(x)
    x_min *= 1 - p_min
    return (x - x_min) / (x_max - x_min)


def lesinn_score(incoming_data: np.array, historical_data: np.array):
    """
    Sampling confidence 
    :param incoming_data: matrix shape=(n,d) n samples, d dimensions 
    :param historical_data: matrix shape=(m,d), m time steps, d dimensions
    :return: Sampling confidence score shape=(n,)
    """
    from algorithm.lesinn import online_lesinn
    return p_normalize(1 / online_lesinn(
        incoming_data, historical_data, random_state=random_state, t=lesinn_t,
        phi=lesinn_phi
    ))


def moving_average_score(
        incoming_data: np.array, historical_data: np.array
):
    """
    Moving average score
    :param incoming_data: matrix shape=(n,d), n samples, d dimensions
    :param historical_data: matrix shape=(m,d), m time steps, d dimensions
    :return: Sampling confidence score shape=(n,)
    """
    from algorithm.moving_average import online_moving_average
    return p_normalize(1 / (1 + online_moving_average(
        incoming_data,
        historical_data,
        moving_average_window,
        moving_average_stride
    )))

 


def read_config(config: dict):
    """
    init global parameters  
    :param config: config dictionary, please refer to detector-config.yml
    :return:
    """
    global random_state, \
        workers, \
        lesinn_t, lesinn_phi, \
        moving_average_window, moving_average_stride, \
        rec_window, rec_stride, \
        det_window, det_stride, \
        anomaly_scoring, \
        sample_score_method, \
        cluster_threshold, \
        sample_rate, \
        latest_windows, \
        scale, rho, sigma, \
        retry_limit, \
        without_grouping, without_localize_sampling, \
        data_path, rb, re, cb, ce, header, rec_windows_per_cycle, \
        label_path, save_path, \
        anomaly_score_example_percentage, anomaly_distance_topn
    if 'sample_score_method' in config.keys():
        sample_score_config = config['sample_score_method']
        if 'lesinn' in sample_score_config.keys():
            lesinn_t = int(sample_score_config['lesinn']['t'])
            lesinn_phi = int(sample_score_config['lesinn']['phi'])
        if 'moving_average' in sample_score_config.keys():
            moving_average_window = \
                int(sample_score_config['moving_average']['window'])
            moving_average_stride = \
                int(sample_score_config['moving_average']['stride'])
    if 'anomaly_scoring' in config.keys():
        anomaly_scoring_config = config['anomaly_scoring']
        if 'anomaly_score_example' in anomaly_scoring_config.keys():
            anomaly_score_example_percentage = \
                int(anomaly_scoring_config['anomaly_score_example']
                    ['percentage'])
            if anomaly_score_example_percentage > 100 \
                    or anomaly_score_example_percentage < 0:
                raise Exception('percentage must be between 0 and 100')
            anomaly_distance_topn = \
                int(anomaly_scoring_config['anomaly_score_example']['topn'])

    if 'global' in config.keys():
        global_config = config['global']
        if 'random_state' in global_config.keys():
            global random_state
            random_state = int(global_config['random_state'])

    data_config = config['data']

    if 'reconstruct' in data_config.keys():
        data_rec_config = data_config['reconstruct']
        rec_window = int(data_rec_config['window'])
        rec_stride = int(data_rec_config['stride'])

    if 'detect' in data_config.keys():
        data_det_config = data_config['detect']
        det_window = int(data_det_config['window'])
        det_stride = int(data_det_config['stride'])

    data_path = data_config['path']
    label_path = data_config['label_path']
    save_path = data_config['save_path']

    header = data_config['header']
    rb, re = data_config['row_begin'], data_config['row_end']
    cb, ce = data_config['col_begin'], data_config['col_end']
    rec_windows_per_cycle = data_config['rec_windows_per_cycle']

    detector_config = config['detector_arguments']
    if detector_config['anomaly_scoring'] == 'anomaly_score_example':
        anomaly_scoring = anomaly_score_example
    else:
        raise Exception(
            'unknown config[detector][anomaly_scoring]: %s',
            detector_config['anomaly_scoring']
        )
    if detector_config['sample_score_method'] == 'lesinn_score':
        sample_score_method = lesinn_score
    elif detector_config['sample_score_method'] == 'moving_average_score':
        sample_score_method = moving_average_score
    else:
        raise Exception(
            'unknown config[detector][sample_score_method]: %s',
            detector_config['sample_score_method']
        )
    workers = int(detector_config['workers'])
    cluster_threshold = float(detector_config['cluster_threshold'])
    sample_rate = float(detector_config['sample_rate'])
    latest_windows = int(detector_config['latest_windows'])
    scale = float(detector_config['scale'])
    rho = float(detector_config['rho'])
    sigma = float(detector_config['sigma'])
    retry_limit = int(detector_config['retry_limit'])
    without_grouping = detector_config['without_grouping']
    without_localize_sampling = bool(
        detector_config['without_localize_sampling']
    )


def run(data: pd.DataFrame):
    """
    :param data input
    """

    n, d = data.shape
    if n < rec_window * rec_windows_per_cycle:
        raise Exception('data point count less than 1 cycle')

    data = data.values
    for i in range(d):
        data[:, i] = normalization(data[:, i])

    print(
        'expected samples per sample unit (recommended >10):',
        np.sqrt(2 * np.pi) * rho * scale * sigma * rec_window
    )

    detector = CSAnomalyDetector(
        workers=workers,
        cluster_threshold=cluster_threshold,
        sample_rate=sample_rate,
        sample_score_method=sample_score_method,
        distance=anomaly_scoring,
        scale=scale,
        rho=rho,
        sigma=sigma,
        random_state=random_state,
        retry_limit=retry_limit,
        without_grouping=without_grouping,
        without_localize_sampling=without_localize_sampling
    )
    rec, retries = detector.reconstruct(
        data, rec_window, rec_windows_per_cycle, rec_stride
    )
    score = detector.predict(
        data, rec, det_window, det_stride
    )
    print('retries:', retries)
    np.savetxt(save_path + '_rec.txt', rec, '%.6f', ',')
    np.savetxt(save_path + '_score.txt', score, '%.6f', ',')
    label = np.loadtxt(label_path, int, delimiter=',')
    
    # # Online choosing threshold 
    # proba = sliding_anomaly_predict(score)
    # precision, recall, f1score = evaluate_result(proba, label[rb:re])

    # Best F-score
    precision, recall, f1score, _ = evaluation(label[rb:re], score)

    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1_score: ', f1score)


if __name__ == '__main__':
    import argparse
    import sys
    import os

    sys.path.append(os.path.abspath('..'))

    from detector import CSAnomalyDetector
    from utils import normalization
    from utils.metrics import sliding_anomaly_predict, evaluate_result, evaluation

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config path')
    args = parser.parse_args()
    config = 'detector-config.yml'
    if args.config:
        config = args.config
    with open(config, 'r', encoding='utf8') as file:
        config_dict = yaml.load(file)
    read_config(config_dict)
    
    run(pd.read_csv(data_path, header=header).iloc[rb:re, cb:ce])
