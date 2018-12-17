from unittest import mock

import pytest
import torch

from dropblock import DropBlock2D


# noinspection PyCallingNonCallable
def test_block_mask_square_even():
    db = DropBlock2D(block_size=2, drop_prob=0.1)
    mask = torch.tensor([[[1., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]])

    expected = torch.tensor([[[0., 0., 1., 1., 1.],
                              [0., 0., 1., 0., 0.],
                              [1., 1., 1., 0., 0.],
                              [1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1.]]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


# noinspection PyCallingNonCallable
def test_block_mask_Hw_even():
    db = DropBlock2D(block_size=2, drop_prob=0.1)
    mask = torch.tensor([[[1., 0., 0., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.]]])

    expected = torch.tensor([[[0., 0., 1., 1.],
                              [0., 0., 1., 0.],
                              [1., 1., 1., 0.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


# noinspection PyCallingNonCallable
def test_block_mask_hW_even():
    db = DropBlock2D(block_size=2, drop_prob=0.1)
    mask = torch.tensor([[[0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]])

    expected = torch.tensor([[[1., 1., 1., 0., 0.],
                              [1., 1., 1., 0., 0.],
                              [1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1.]]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


# noinspection PyCallingNonCallable
def test_block_mask_square_odd():
    db = DropBlock2D(block_size=3, drop_prob=0.1)
    mask = torch.tensor([[[1., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]])

    expected = torch.tensor([[[0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0.],
                              [1., 1., 0., 0., 0.],
                              [1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1.]]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


# noinspection PyCallingNonCallable
def test_block_mask_Hw_odd():
    db = DropBlock2D(block_size=3, drop_prob=0.1)
    mask = torch.tensor([[[1., 0., 0., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.],
                          [0., 0., 0., 0.]]])

    expected = torch.tensor([[[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [1., 1., 0., 0.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


# noinspection PyCallingNonCallable
def test_block_mask_hW_odd():
    db = DropBlock2D(block_size=3, drop_prob=0.1)
    mask = torch.tensor([[[0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]])

    expected = torch.tensor([[[1., 1., 0., 0., 0.],
                              [1., 1., 0., 0., 0.],
                              [1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1.]]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


# noinspection PyCallingNonCallable
def test_block_mask_overlap():
    db = DropBlock2D(block_size=2, drop_prob=0.1)
    mask = torch.tensor([[[1., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0.]]])

    expected = torch.tensor([[[0., 0., 1., 1., 1.],
                              [0., 0., 0., 1., 1.],
                              [1., 0., 0., 1., 1.],
                              [1., 1., 1., 1., 1.],
                              [1., 1., 1., 1., 1.]]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


# noinspection PyCallingNonCallable
def test_forward_pass():
    db = DropBlock2D(block_size=3, drop_prob=0.1)
    block_mask = torch.tensor([[[0., 0., 0., 1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 0., 1.],
                                [0., 0., 0., 0., 0., 0., 1.],
                                [1., 1., 1., 0., 0., 0., 1.],
                                [1., 1., 1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 1., 1., 1.]]])

    db._compute_block_mask = mock.MagicMock(return_value=block_mask)

    x = torch.ones(10, 10, 7, 7)
    h = db(x)

    expected = block_mask * block_mask.numel() / block_mask.sum()
    expected = expected[:, None, :, :].expand_as(x)

    assert tuple(h.shape) == (10, 10, 7, 7)
    assert torch.equal(h, expected)


def test_forward_pass2():

    block_sizes = [2, 3, 4, 5, 6, 7, 8]
    heights = [5, 6, 8, 10, 11, 14, 15]
    widths = [5, 7, 8, 10, 15, 14, 15]

    for block_size, height, width in zip(block_sizes, heights, widths):
        dropout = DropBlock2D(0.1, block_size=block_size)
        input = torch.randn((5, 20, height, width))
        output = dropout(input)
        assert tuple(input.shape) == tuple(output.shape)


def test_large_block_size():
    dropout = DropBlock2D(0.3, block_size=9)
    x = torch.rand(100, 10, 16, 16)
    output = dropout(x)

    assert tuple(x.shape) == tuple(output.shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_pass_with_cuda():
    dropout = DropBlock2D(0.3, block_size=5).to('cuda')
    x = torch.rand(100, 10, 16, 16).to('cuda')
    output = dropout(x)

    assert tuple(x.shape) == tuple(output.shape)
