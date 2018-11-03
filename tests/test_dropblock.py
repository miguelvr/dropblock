import torch
from dropblock import DropBlock
from unittest import mock


# noinspection PyCallingNonCallable
def test_block_mask_no_overlap():
    db = DropBlock(block_size=2, feat_size=10, drop_prob=0.1)
    mask = torch.tensor([[1., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]])

    expected = torch.tensor([[0., 0., 1., 1., 1.],
                             [0., 0., 1., 0., 0.],
                             [1., 1., 1., 0., 0.],
                             [1., 1., 1., 1., 1.],
                             [1., 1., 1., 1., 1.]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


# noinspection PyCallingNonCallable
def test_block_mask_overlap():
    db = DropBlock(block_size=3, feat_size=10, drop_prob=0.1)
    mask = torch.tensor([[1., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0.]])

    expected = torch.tensor([[0., 0., 0., 1., 1.],
                             [0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0.],
                             [1., 1., 1., 0., 0.],
                             [1., 1., 1., 1., 1.]])

    block_mask = db._compute_block_mask(mask)
    assert torch.equal(block_mask, expected)


def test_forward_pass():
    db = DropBlock(block_size=3, feat_size=10, drop_prob=0.1)
    block_mask = torch.tensor([[0., 0., 1., 1., 1.],
                               [0., 0., 1., 0., 0.],
                               [1., 1., 1., 0., 0.],
                               [1., 1., 1., 1., 1.],
                               [1., 1., 1., 1., 1.]])

    db._compute_block_mask = mock.MagicMock(return_value=block_mask)

    x = torch.ones(10, 10, 5, 5)
    h = db(x)

    expected = block_mask * block_mask.numel() / block_mask.sum()
    expected = expected[None, None, :, :].expand_as(x)

    assert tuple(h.shape) == (10, 10, 5, 5)
    assert torch.equal(h, expected)
