import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(feat_size=x.shape[1])

            # sample from a mask
            mask_reduction = self.block_size // 2
            mask_height = x.shape[2] - mask_reduction
            mask_width = x.shape[3] - mask_reduction
            mask = Bernoulli(gamma).sample((x.shape[0], mask_height, mask_width))

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.conv2d(mask[:, None, :, :],
                              torch.ones((mask.shape[0], 1, self.block_size, self.block_size)),
                              padding=self.block_size // 2 + 1)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, 1:, 1:]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, feat_size):
        return (self.drop_prob / (self.block_size ** 2)) * \
               ((feat_size ** 2) / ((feat_size - self.block_size + 1) ** 2))


class DropBlock3D(DropBlock2D):

    def forward(self, x):
        # shape: (bsize, channels, depth, height, width)

        assert x.dim() == 5, \
            "Expected input with 5 dimensions (bsize, channels, depth, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(feat_size=x.shape[1])

            # sample from a mask
            mask_reduction = self.block_size // 2
            mask_depth = x.shape[-3] - mask_reduction
            mask_height = x.shape[-2] - mask_reduction
            mask_width = x.shape[-1] - mask_reduction
            mask = Bernoulli(gamma).sample((x.shape[0], mask_depth, mask_height, mask_width))

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.conv3d(mask[:, None, :, :, :],
                              torch.ones((mask.shape[0], 1, self.block_size, self.block_size, self.block_size)),
                              padding=self.block_size // 2 + 1)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, 1:, 1:, 1:]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask
