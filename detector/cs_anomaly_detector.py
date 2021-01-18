import numpy as np
import os
from algorithm.cluster import cluster
from algorithm.cvxpy import reconstruct
from algorithm.sampling import localized_sample
from cvxpy.error import SolverError
from multiprocessing import Process, Event, Queue
from threading import Thread

max_seed = 10 ** 9 + 7


class CycleFeatureProcess(Process):
    """
    计算单个周期内特征的线程
    """

    def __init__(
            self,
            task_queue: Queue,
            result_queue: Queue,
            cluster_threshold: float
    ):
        """
        :param task_queue: 作业队列
        :param result_queue: 结果队列
        :param cluster_threshold: 聚类阈值
        """
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.cluster_threshold = cluster_threshold

    def run(self):
        print('CycleFeatureProcess-%d: start' % os.getpid())
        while not self.task_queue.empty():
            group_index, cycle_data = self.task_queue.get()
            self.result_queue.put(
                (group_index, cluster(cycle_data, self.cluster_threshold))
            )
            print(
                'CycleFeatureProcess-%d: finish task-%d' %
                (os.getpid(), group_index)
            )
        print('CycleFeatureProcess-%d: exit' % os.getpid())


class WindowReconstructProcess(Process):
    """
    窗口重建工作进程
    """

    def __init__(
            self,
            data: np.array,
            task_queue: Queue,
            result_queue: Queue,
            cycle: int,
            latest_windows: int,
            sample_score_method,
            sample_rate: float,
            scale: float,
            rho: float,
            sigma: float,
            random_state: int,
            without_localize_sampling: bool,
            retry_limit: int,
            task_return_event: Event()
    ):
        """
        :param data: 原始数据的拷贝
        :param task_queue: 作业队列
        :param result_queue: 结果队列
        :param cycle: 周期
        :param latest_windows: 计算采样价值指标时参考的最近历史周期数
        :param sample_score_method: 计算采样价值指标方法
        :param sample_rate: 采样率
        :param scale: 采样参数: 等距采样点扩充倍数
        :param rho: 采样参数: 中心采样概率
        :param sigma: 采样参数: 采样集中程度
        :param random_state: 随机数种子
        :param without_localize_sampling: 是否不按局部化采样算法进行采样
        :param retry_limit: 每个窗口重试的上限
        :param task_return_event: 当一个作业被完成时触发的事件, 通知主进程收集
        """
        super().__init__()
        self.data = data
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.cycle = cycle
        self.latest_windows = latest_windows
        self.sample_score_method = sample_score_method
        self.sample_rate = sample_rate
        self.without_localize_sampling = without_localize_sampling
        self.scale = scale
        self.rho = rho
        self.sigma = sigma
        self.random_state = random_state
        self.retry_limit = retry_limit
        self.task_return_event = task_return_event

    def run(self):
        from time import time
        if self.random_state:
            np.random.seed(self.random_state)
        tot = time()
        data_process = 0
        wait_syn = 0
        rec = 0
        sample_scoring = 0
        print('WindowReconstructProcess-%d: start' % os.getpid())
        while not self.task_queue.empty():
            t = time()
            wb, we, group = self.task_queue.get()
            wait_syn += time() - t
            t = time()
            hb = max(0, wb - self.latest_windows)
            latest = self.data[hb:wb]
            window_data = self.data[wb:we]
            data_process += time() - t
            t = time()
            sample_score = self.sample_score_method(window_data, latest)
            sample_scoring += time() - t
            t = time()
            rec_window, retries = \
                self.window_sample_reconstruct(
                    data=window_data,
                    groups=group,
                    score=sample_score,
                    random_state=self.random_state * wb * we % max_seed
                )
            rec += time() - t
            t = time()
            self.result_queue.put((wb, we, rec_window, retries, sample_score))
            self.task_return_event.set()
            wait_syn += time() - t
            print(
                'WindowReconstructProcess-%d: window[%d, %d) done' %
                (os.getpid(), wb, we)
            )
        tot = time() - tot
        print('WindowReconstructProcess-%d: exit' % os.getpid())
        print(
            'tot: %f\ndata_process: %f\nwait_syn: %f\nrec: %f\n'
            'sample_scoring:%f\n'
            % (tot, data_process, wait_syn, rec, sample_scoring)
        )

    def sample(self, x: np.array, m: int, score: np.array, random_state: int):
        """
        取得采样的数据
        :param x: kpi等距时间序列, shape=(n,d), n是行数, d是维度
        :param m: 采样个数
        :param score: 采样点置信度
        :param random_state: 采样随机种子
        :return: 采样序列数组X, X[i][0]是0~n-1的实数, 表示该采样点的时间点,
                X[i][1] 是shape=(k,)的数组, 表示该时间点各个维度kpi数据  0<i<m
                已经按X[i][0]升序排序
        """
        n, d = x.shape
        data_mat = np.mat(x)
        sample_matrix, timestamp = localized_sample(
            x=data_mat, m=m,
            score=score,
            scale=self.scale, rho=self.rho, sigma=self.sigma,
            random_state=random_state
        )
        # 采样中心对应的位置
        s = np.array(sample_matrix * data_mat)
        res = []
        for i in range(m):
            res.append((timestamp[i], s[i]))
        res.sort(key=lambda each: each[0])
        res = np.array(res)
        timestamp = np.array(res[:, 0]).astype(int)
        values = np.zeros((m, d))
        for i in range(m):
            values[i, :] = res[i][1]
        return timestamp, values

    def window_sample_reconstruct(
            self,
            data: np.array,
            groups: list,
            score: np.array,
            random_state: int
    ):
        """
        :param data: 原始数据
        :param groups: 分组
        :param score: 这个窗口的每一个点的采样可信度
        :param random_state: 随机种子
        :return: 重建数据, 重建尝试次数
        """
        # 数据量, 维度
        n, d = data.shape
        retry_count = 0
        sample_rate = self.sample_rate
        while True:
            try:
                if self.without_localize_sampling:
                    if random_state:
                        np.random.seed(random_state)
                    timestamp = np.random.choice(
                        np.arange(n),
                        size=int(np.round(sample_rate * n)),
                        replace=False
                    )
                    values = data[timestamp]
                else:
                    timestamp, values = \
                        self.sample(
                            data,
                            int(np.round(sample_rate * n)),
                            score,
                            random_state
                        )
                rec = np.zeros(shape=(n, d))
                for i in range(len(groups)):
                    x_re = reconstruct(
                        n, len(groups[i]), timestamp,
                        values[:, groups[i]]
                    )
                    for j in range(len(groups[i])):
                        rec[:, groups[i][j]] = x_re[:, j]
                break
            except SolverError:
                if retry_count > self.retry_limit:
                    raise Exception(
                        'retry failed, please try higher sample rate or '
                        'window size'
                    )
                sample_rate += (1 - sample_rate) / 4
                retry_count += 1
                from sys import stderr
                stderr.write(
                    'WARNING: reconstruct failed, retry with higher '
                    'sample rate %f, retry times remain %d\n'
                    % (
                        sample_rate, self.retry_limit - retry_count)
                )
        return rec, retry_count


class CSAnomalyDetector:
    """
    基于压缩感知采样重建的离线多进程异常检测器
    """

    def __init__(
            self,
            cluster_threshold: float,
            sample_rate: float,
            sample_score_method,
            distance,
            workers: int = 1,
            latest_windows: int = 96,
            scale: float = 5,
            rho: float = 0.1,
            sigma: float = 1 / 24,
            random_state=None,
            retry_limit=10,
            without_grouping: str = None,
            without_localize_sampling: bool = False,
    ):
        """
        :param cluster_threshold: 聚类参数: 阈值
        :param sample_rate: 采样率
        :param sample_score_method: 采样点可信度计算函数 输入(array(n * d))输出
                array(n) 表示输入的n个点的采样可信度
        :param distance: 计算距离的函数, 输入 (array(n * d), array(n * d)) 输出
            real表示两个输入之间的距离
        :param workers: 计算线程数
        :param latest_windows: 采样时参考的历史窗口数
        :param scale: 采样参数: 等距采样点扩充倍数
        :param rho: 采样参数: 中心采样概率
        :param sigma: 采样参数: 采样集中程度
        :param random_state: 随机数种子
        :param retry_limit: 求解重试次数, 超过次数求解仍未成功, 则抛出异常
        :param without_grouping: 降级实验: 不进行分组
        :param without_localize_sampling: 降级实验: 完全随机采样
        """
        if sample_rate > 1 or sample_rate <= 0:
            raise Exception('invalid sample rate: %s' % sample_rate)
        if without_grouping and without_grouping not in \
                {'one_by_one', 'all_by_one'}:
            raise Exception('unknown without grouping option')
        self._scale = scale
        self._rho = rho
        self._sigma = sigma
        self._sample_rate = sample_rate
        self._cluster_threshold = cluster_threshold
        self._random_state = random_state
        self._latest_windows = latest_windows
        # 采样点可信度计算方法
        self._sample_score_method = sample_score_method
        # 距离计算方法
        self._distance = distance
        # 重试参数
        self._retry_limit = retry_limit
        # 最大工作线程数
        self._workers = workers
        # 降级实验
        self._without_grouping = without_grouping
        self._without_localize_sampling = without_localize_sampling

    def reconstruct(
            self, data: np.array,
            window: int,
            windows_per_cycle: int,
            stride: int = 1,
    ):
        """
        离线预测输入数据的以时间窗为单位的异常概率预测, 多线程
        :param data: 输入数据
        :param window: 时间窗口长度(点)
        :param windows_per_cycle: 周期长度: 以时间窗口为单位
        :param stride: 时间窗口步长
        """
        if windows_per_cycle < 1:
            raise Exception('a cycle contains 1 window at least')
        # 周期长度
        cycle = windows_per_cycle * window
        # 周期特征: 按周期分组
        groups = self._get_cycle_feature(data, cycle)
        print('group per cycle:')
        for i in range(len(groups)):
            print('cycle: %d ----' % i)
            for each in groups[i]:
                print('  ', each)
        reconstructed, retry_count = self._get_reconstructed_data(
            data, window, windows_per_cycle, groups, stride)
        return reconstructed, retry_count

    def predict(
            self,
            data: np.array,
            reconstructed: np.array,
            window: int,
            stride: int = 1,
    ):
        """
        离线处理: 利用参数进行评估, 得到每个点的异常得分
        :param data: 原始数据
        :param reconstructed: 重建好的数据
        :param window: 数据窗口长度
        :param stride: 窗口步长
        :return: 每个点的异常得分
        """
        if reconstructed.shape != data.shape:
            raise Exception('shape mismatches')
        n, d = data.shape
        # 异常得分
        anomaly_score = np.zeros((n,))
        # 表示当时某个位置上被已重建窗口的数量
        anomaly_score_weight = np.zeros((n,))
        # 窗口左端点索引
        wb = 0
        while True:
            we = min(n, wb + window)
            # 窗口右端点索引 窗口数据[wb, we)
            score = self._distance(data[wb:we], reconstructed[wb:we])
            for i in range(we - wb):
                w = i + wb
                weight = anomaly_score_weight[w]
                anomaly_score[w] = \
                    (anomaly_score[w] * weight + score) / (weight + 1)
            anomaly_score_weight[wb:we] += 1
            if we >= n:
                break
            wb += stride
        return anomaly_score

    def _get_reconstructed_data(
            self,
            data: np.array,
            window: int,
            windows_per_cycle: int,
            groups: list,
            stride: int,
    ):
        """
        离线预测输入数据的以时间窗为单位的异常概率预测, 多线程
        :param data: 输入数据
        :param window: 时间窗口长度(点)
        :param windows_per_cycle: 周期长度: 以时间窗口为单位
        :param groups: 每个周期的分组
        :param stride: 时间窗口步长
        :return:
        """
        n, d = data.shape
        # 重建的数据
        reconstructed = np.zeros((n, d))
        # 表示当时某个位置上被已重建窗口的数量
        reconstructing_weight = np.zeros((n,))
        needed_weight = np.zeros((n,))
        # 作业列表
        task_queue = Queue()
        # 结果列表
        result_queue = Queue()
        # 周期长度
        cycle = window * windows_per_cycle

        # 窗口左端点索引
        win_l = 0
        while True:
            win_r = min(n, win_l + window)
            # 窗口右端点索引 窗口数据[win_l, win_r)
            task_queue.put((win_l, win_r, groups[win_l // cycle]))
            needed_weight[win_l:win_r] += 1
            if win_r >= n:
                break
            win_l += stride

        task_return_event = Event()
        finished = False

        def receive_result_thread():
            """
            接受result_queue结果的线程
            :return:
            """
            total_retries = 0
            while True:
                while result_queue.empty():
                    task_return_event.clear()
                    task_return_event.wait()
                    if finished:
                        result_queue.put(total_retries)
                        return
                wb, we, rec_window, retries, sample_score = result_queue.get()
                total_retries += retries
                for index in range(rec_window.shape[0]):
                    w = index + wb
                    weight = reconstructing_weight[w]
                    reconstructed[w, :] = \
                        (reconstructed[w, :] * weight + rec_window[index]) \
                        / (weight + 1)
                reconstructing_weight[wb:we] += 1

        processes = []
        for i in range(self._workers):
            process = WindowReconstructProcess(
                data=data, task_queue=task_queue, result_queue=result_queue,
                cycle=cycle, latest_windows=self._latest_windows,
                sample_score_method=self._sample_score_method,
                sample_rate=self._sample_rate,
                scale=self._scale, rho=self._rho, sigma=self._sigma,
                random_state=self._random_state,
                without_localize_sampling=self._without_localize_sampling,
                retry_limit=self._retry_limit,
                task_return_event=task_return_event
            )
            process.start()
            processes.append(process)
        receiving_thread = Thread(target=receive_result_thread)
        receiving_thread.start()
        for each in processes:
            each.join()
        finished = True
        task_return_event.set()
        receiving_thread.join()

        mismatch_weights = []
        for i in range(n):
            if reconstructing_weight[i] != needed_weight[i]:
                mismatch_weights.append('%d' % i)
        if len(mismatch_weights):
            from sys import stderr
            stderr.write('BUG empty weight: index: %s\n' %
                         ','.join(mismatch_weights))
        return reconstructed, result_queue.get()

    def _get_cycle_feature(
            self,
            data: np.array,
            cycle: int,
    ):
        """
        将数据按周期进行划分后计算得到每个周期的分组
        :param data: 数据
        :param cycle: 周期长度
        :return: 分组结果
        """
        # 数据量, 维度
        n, d = data.shape
        # 每周期分组结果
        cycle_groups = []
        # 工作数量
        group_index = 0
        # 作业队列, 用于向子进程输入数据
        task_queue = Queue()
        # 输出队列
        result_queue = Queue()
        # 周期开始的index
        cb = 0
        while cb < n:
            # 周期结束的index
            ce = min(n, cb + cycle)  # 一周期数据为data[cb, ce)
            # 初始化追加列表引用
            if group_index == 0:
                # 没有历史数据
                # 分组默认每个kpi一组
                init_group = []
                if not self._without_grouping:
                    for i in range(d):
                        init_group.append([i])
                cycle_groups.append(init_group)
            else:
                cycle_groups.append([])
                # 向工作队列中填充输入数据
                if not self._without_grouping:
                    task_queue.put((group_index, data[cb:ce]))
            group_index += 1
            cb += cycle
        if self._without_grouping:
            if self._without_grouping == 'one_by_one':
                # 每条kpi一组
                for each in cycle_groups:
                    for i in range(d):
                        each.append([i])
            elif self._without_grouping == 'all_by_one':
                # 所有kpi一组
                all_in_group = []
                for i in range(d):
                    all_in_group.append(i)
                for each in cycle_groups:
                    each.append(all_in_group)
        else:
            processes = []
            for i in range(min(len(cycle_groups), self._workers)):
                process = CycleFeatureProcess(
                    task_queue, result_queue, self._cluster_threshold
                )
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
            while not result_queue.empty():
                group_index, group = result_queue.get()
                cycle_groups[group_index] = group
        return cycle_groups
