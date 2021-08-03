# Jump-Starting Multivariate Time Series Anomaly Detection (JumpStarter)

JumpStarter is a comprehensive multivariate time series anomaly detection approach based on **Compressed Sensing** (CS). CS is a signal processing technique where high-energy components in a matrix (multivariate time series) are sparse (i.e. have few high-energy components).
Hence, the difference between the original and the reconstructed multivariate time series, comprised only of low-energy components, should resemble white noise, when the original time series contains no anomaly.
The intuition behind using CS for anomaly detection is that anomalies in multivariate time series, such as jitters, sudden drops or surges, usually manifest themselves as strong signals that contain high-energy components, which would differ significantly from white noise. 
Hence we can tell whether a time series contains anomalies by checking whether the difference between the original and the reconstructed multivariate time series in a sliding window looks very differently from white noise.


## API Demo Usage

```
cd detector
python run_detector.py
```

## Datasets

[Dataset1](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset) is collected from a large Internet company _A_. 

Dataset2 and Dataset3 are collected from a top-tier global content platform _B_ providing services for over 800 million daily active (over 1 billion cumulative) users across all of its content platforms.
    
|Dataset|# Services|# Metrics|# Training Days|# Test Days|Anomaly Ratio|  
|----|----|----|----|----|----|
|Dataset1|28|38|13|13|4.16|
|Dataset2|30|19|20|25|5.25|
|Dataset3|30|19|20|25|20.26|

More details can be found in our [paper](https://www.usenix.org/conference/atc21/presentation/ma). 

## Citing JumpStarter

JumpStarter paper is published in USENIX ATC. If you use JumpStarter in a scientific publication, we would appreciate citations to the following paper:
```
Ma, M., Zhang, S. and Chen, J., et.al. Jump-Starting Multivariate Time Series Anomaly Detection for Online Service Systems. USENIX ATC 2021.
```
BibTex:
```
@inproceedings{ma2021jump,
  title={Jump-Starting Multivariate Time Series Anomaly Detection for Online Service Systems},
  author={Ma, Minghua and Zhang, Shenglin and Chen, Junjie and Xu, Jim and Li, Haozhe and Lin, Yongliang and Nie, Xiaohui and Zhou, Bo and Wang, Yong and Pei, Dan},
  booktitle={2021 $\{$USENIX$\}$ Annual Technical Conference ($\{$USENIX$\}$$\{$ATC$\}$ 21)},
  pages={413--426},
  year={2021}
}
```
