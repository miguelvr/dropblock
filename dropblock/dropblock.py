import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropBlock(nn.Module):
    def __init__(self, block_size, gamma):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        self.gamma = gamma
        self.bernouli = Bernoulli(gamma)

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        if self.training:
            return x
        else:
            mask = self.bernouli.sample((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)

    def _compute_block_mask(self, mask):
        height, width = mask.shape

        non_zero_idxs = mask.nonzero()
        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size)
            ]
        ).t()

        non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
        offsets = offsets.repeat(1, self.block_size).view(-1, 2)

        block_idxs = non_zero_idxs + offsets
        padded_mask = F.pad(mask, (0, self.block_size, 0, self.block_size))

        padded_mask[block_idxs[:, 0], block_idxs[:, 1]] = 1.
        block_mask = padded_mask[:height, :width]

        return block_mask
