import torch
import torch.nn.functional as F


if torch.__version__ >= '1.0.0':
    Module = torch.jit.ScriptModule
    script_method = torch.jit.script_method
else:
    Module = torch.nn.Module

    def script_method(x):
        return x


class DropBlock2D(Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.

    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    __constants__ = ['drop_prob', 'block_size', 'gamma']

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob  # type: float
        self.block_size = block_size  # type: int

        # get gamma value
        self.gamma = self._compute_gamma()  # type: float

    @script_method
    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            out = x
        else:
            # sample mask
            mask = (torch.rand(x.shape[0], x.shape[2], x.shape[3], device=x.device) < self.gamma).float()

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask.unsqueeze(1)

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

        return out

    @script_method
    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask.unsqueeze(1),
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self):
        return self.drop_prob / (self.block_size ** 2)


class DropBlock3D(Module):
    r"""Randomly zeroes 3D spatial blocks of the input tensor.

    An extension to the concept described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop

    Shape:
        - Input: `(N, C, D, H, W)`
        - Output: `(N, C, D, H, W)`

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890

    """

    __constants__ = ['drop_prob', 'block_size', 'gamma']

    def __init__(self, drop_prob, block_size):
        super(DropBlock3D, self).__init__()

        self.drop_prob = drop_prob  # type: float
        self.block_size = block_size  # type: int

        # get gamma value
        self.gamma = self._compute_gamma()  # type: float

    @script_method
    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            out = x
        else:
            # sample mask
            mask = (torch.rand(x.shape[0], x.shape[2], x.shape[3], x.shape[4], device=x.device) < self.gamma).float()

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask.unsqueeze(1)

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

        return out

    @script_method
    def _compute_block_mask(self, mask):
        block_mask = F.max_pool3d(input=mask.unsqueeze(1),
                                  kernel_size=(self.block_size, self.block_size, self.block_size),
                                  stride=(1, 1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self):
        return self.drop_prob / (self.block_size ** 3)
