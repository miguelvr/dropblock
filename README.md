# DropBlock

![build](https://travis-ci.org/miguelvr/dropblock.png?branch=master)
[![Downloads](https://pepy.tech/badge/dropblock)](https://pepy.tech/project/dropblock)


Implementation of [DropBlock: A regularization method for convolutional networks](https://arxiv.org/pdf/1810.12890.pdf) 
in PyTorch.

## Abstract

Deep neural networks often work well when they are over-parameterized 
and trained with a massive amount of noise and regularization, such as 
weight decay and dropout. Although dropout is widely used as a regularization 
technique for fully connected layers, it is often less effective for convolutional layers. 
This lack of success of dropout for convolutional layers is perhaps due to the fact 
that activation units in convolutional layers are spatially correlated so 
information can still flow through convolutional networks despite dropout. 
Thus a structured form of dropout is needed to regularize convolutional networks. 
In this paper, we introduce DropBlock, a form of structured dropout, where units in a 
contiguous region of a feature map are dropped together. 
We found that applying DropBlock in skip connections in addition to the 
convolution layers increases the accuracy. Also, gradually increasing number 
of dropped units during training leads to better accuracy and more robust to hyperparameter choices. 
Extensive experiments show that DropBlock works better than dropout in regularizing 
convolutional networks. On ImageNet classification, ResNet-50 architecture with 
DropBlock achieves 78.13% accuracy, which is more than 1.6% improvement on the baseline. 
On COCO detection, DropBlock improves Average Precision of RetinaNet from 36.8% to 38.4%.


## Installation

Install directly from PyPI:

    pip install dropblock
    
or the bleeding edge version from github:

    pip install git+https://github.com/miguelvr/dropblock.git#egg=dropblock

**NOTE**: Implementation and tests were done in Python 3.6, if you have problems with other versions of python please open an issue.

## Usage


For 2D inputs (DropBlock2D):

```python
import torch
from dropblock import DropBlock2D

# (bsize, n_feats, height, width)
x = torch.rand(100, 10, 16, 16)

drop_block = DropBlock2D(block_size=3, drop_prob=0.3)
regularized_x = drop_block(x)
```

For 3D inputs (DropBlock3D):

```python
import torch
from dropblock import DropBlock3D

# (bsize, n_feats, depth, height, width)
x = torch.rand(100, 10, 16, 16, 16)

drop_block = DropBlock3D(block_size=3, drop_prob=0.3)
regularized_x = drop_block(x)
```

Scheduled Dropblock:

```python
import torch
from dropblock import DropBlock2D, LinearScheduler

# (bsize, n_feats, depth, height, width)
loader = [torch.rand(20, 10, 16, 16) for _ in range(10)]

drop_block = LinearScheduler(
                DropBlock2D(block_size=3, drop_prob=0.),
                start_value=0.,
                stop_value=0.25,
                nr_steps=5
            )

probs = []
for x in loader:
    drop_block.step()
    regularized_x = drop_block(x)
    probs.append(drop_block.dropblock.drop_prob)
    
print(probs)
```

The drop probabilities will be:
```
>>> [0.    , 0.0625, 0.125 , 0.1875, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
```

The user should include the `step()` call at the start of the batch loop, 
or at the the start of a model's `forward` call. 

Check [examples/resnet-cifar10.py](examples/resnet-cifar10.py) to
see an implementation example.

## Implementation details

We use `drop_prob` instead of `keep_prob` as a matter of preference, 
and to keep the argument consistent with pytorch's dropout. 
Regardless, everything else should work similarly to what is described in the paper.

## Benchmark

Refer to [BENCHMARK.md](BENCHMARK.md)

## Reference
[Ghiasi et al., 2018] DropBlock: A regularization method for convolutional networks

## TODO
- [x] Scheduled DropBlock
- [x] Get benchmark numbers
- [x] Extend the concept for 3D images
