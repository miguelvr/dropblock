# ResNet-9 CIFAR-10 Benchmark


Results for ResNet9 on CIFAR10, trained on 1 x NVidia V100 GPU, average over 3 runs:

| Model                | Accuracy (%) | Time (s) |
|----------------------|--------------|----------|
| ResNet9              | 81.46        | 271      |
| ResNet9 + DropBlock* | 81.65        | 288      |

`* scheduled dropblock with block_size=5 and increasing drop_prob 
from 0.0 to 0.25 over 5000 iterations`

Example available [here](examples/resnet-cifar10.py)
