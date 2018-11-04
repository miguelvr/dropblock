# DropBlock

![build](https://travis-ci.org/miguelvr/dropblock.png?branch=master)


Implementation of [DropBlock: A regularization method for convolutional networks](https://arxiv.org/pdf/1810.12890.pdf) 
in PyTorch.

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
- [ ] Get benchmark numbers
- [ ] Extend the concept for 3D images
