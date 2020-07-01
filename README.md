# Maskflownet.pytorch(updating...)
This is an unoffical implementation of [MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask](https://github.com/microsoft/MaskFlownet) by using Pytorch.

## Introduction
![mask_visualization](./306.png)

## Requirements
- Pytorch 1.5.0(only for using **DeformConv**)
- CUDA 10.1+
- Tensorboard
## Install
### correlation
The correlation package must be installed first:
```
cd model/correlation_package
python setup.py install
```
### DCN
if your pytorch is lower than 1.5.0, please install dcn package:
```
cd model/dcn
python setup.py install
```
