# DropBlock

![build](https://travis-ci.org/miguelvr/dropblock.png?branch=master)


Implementation of [DropBlock: A regularization method for convolutional networks](https://arxiv.org/pdf/1810.12890.pdf) 
in PyTorch.

## Abstract

Deep neural networks often work well when they are over-parameterized and trained with a massive amount of noise and regularization, such as weight decay and dropout. Although dropout is widely used as a regularization technique for fully connected layers, it is often less effective for convolutional layers. This lack of success of dropout for convolutional layers is perhaps due to the fact that activation units in convolutional layers are spatially correlated so information can still flow through convolutional networks despite dropout. Thus a structured form of dropout is needed to regularize convolutional networks. In this paper, we introduce DropBlock, a form of structured dropout, where units in a contiguous region of a feature map are dropped together. We found that applying DropBlock in skip connections in addition to the convolution layers increases the accuracy. Also, gradually increasing number of dropped units during training leads to better accuracy and more robust to hyperparameter choices. Extensive experiments show that DropBlock works better than dropout in regularizing convolutional networks. On ImageNet classification, ResNet-50 architecture with DropBlock achieves 78.13% accuracy, which is more than 1.6% improvement on the baseline. On COCO detection, DropBlock improves Average Precision of RetinaNet from 36.8% to 38.4%.



## Installation

Install directly from PyPI:

    pip install dropblock

## Usage

````python
import torch
from dropblock import DropBlock

# (bsize, n_feats, height, width)
x = torch.rand(100, 10, 16, 16)

drop_block = DropBlock(block_size=4, drop_prob=0.3, feat_size=10)
regularized_x = drop_block(x)
````

## Implementation details

Some implementation details differ slightly from the description in the paper, 
but they should be equivalent in terms of performance:

 - We use `drop_prob` instead of `keep_prob`
    - Because of this decision, out bernoulli distribution 
 generates 1's with `drop_prob` instead of 1's with `keep_prob`
 - The block masks are generated from the top left corner, 
 instead of being generated form the centers
    - Therefore we sample from the whole image, instead of the subset 
    where the blocks will fit.
 
## Reference
[Ghiasi et al., 2018] DropBlock: A regularization method for convolutional networks

## TODO
- [ ] Scheduled DropBlock
- [ ] Get benchmark numbers
- [ ] Extend the concept for 3D images
