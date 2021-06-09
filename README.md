# Jump-Starting Multivariate Time Series Anomaly Detection (JumpStarter)

JumpStarter is a comprehensive multivariate time series anomaly detection approach based on Compressed Sensing. CS is a signal processing technique where high-energy components in a matrix (multivariate time series) are sparse (i.e. have few high-energy components).  
Hence, the difference between the original and the reconstructed multivariate time series, comprised only of low-energy components, should resemble white noise, when the original time series contains no anomaly.
The intuition behind using CS for anomaly detection is that anomalies in multivariate time series, such as jitters, sudden drops or surges, usually manifest themselves as strong signals that contain high-energy components, which would differ significantly from white noise. 
Hence we can tell whether a time series contains anomalies by checking whether the difference between the original and the reconstructed multivariate time series in a sliding window looks very differently from white noise.


## API Demo Usage

```
cd detector
python run_detector.py
```

## Anonymized Datasets

https://cloud.tsinghua.edu.cn/f/e958fb8ec7d14c84abe0/

## Citing JumpStarter: 

JumpStarter paper is published in USENIX ATC. If you use JumpStarter in a scientific publication, we would appreciate citations to the following paper:
```
Ma, M., Zhang, S. and Chen, J., et.al. Jump-Starting Multivariate Time Series Anomaly Detection for Online Service Systems. USENIX ATC 2021.
```
