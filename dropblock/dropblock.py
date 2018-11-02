import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropBlock(nn.Module):
    def __init__(self, drop_prob, block_size, feat_size):
        super(DropBlock, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size
        self.feat_size = feat_size
        self.gamma = self._compute_gamma()
        self.bernouli = Bernoulli(self.gamma)

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if self.training:
            return x
        else:
            mask = self.bernouli.sample((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            out = x * block_mask[None, None, :, :]
            return out * block_mask.numel() / block_mask.sum()

    def _compute_block_mask(self, mask):
        height, width = mask.shape

        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size)
            ]
        ).t()

        non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
        offsets = offsets.repeat(1, nr_blocks).view(-1, 2)

        block_idxs = non_zero_idxs + offsets
        padded_mask = F.pad(mask, (0, self.block_size, 0, self.block_size))

        padded_mask[block_idxs[:, 0], block_idxs[:, 1]] = 1.
        block_mask = padded_mask[:height, :width]

        return 1 - block_mask

    def _compute_gamma(self):
        return (self.drop_prob / (self.block_size ** 2)) * \
               ((self.feat_size ** 2) / ((self.feat_size - self.block_size + 1) ** 2))

    def set_drop_probability(self, drop_prob):
        self.drop_prob = drop_prob
        self.gamma = self._compute_gamma()
        self.bernouli = Bernoulli(self.gamma)
