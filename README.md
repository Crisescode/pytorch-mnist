# pytorch-mnist
Some pytorch demo scripts based on mnist datasets, so that everyone can get started with the pytorch deep learning framework.

# Requirement
* Python3
* PyTorch~=1.0.0
* TorchVision
* Scikit-Image

# Usage

## single cpu
``` language
cd single_cpu
python main.py -b 256 -e 20 
```

## single gpu
``` language
cd single_gpu
python main.py -b 256 -e 20 -g 1 
```

## DataParallel
``` language
cd data_parallel
python main.py -b 256 -e 20 -g 2\3\4
```

## DistributedDataParallel
pass

# Performance

## single gpu / DataParallel
* batch size: 256
* batch time: 6s
* training time: 5min
* gpu util: 99%
* gpu memory: 10G