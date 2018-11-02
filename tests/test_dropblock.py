import torch
from dropblock import DropBlock


# noinspection PyCallingNonCallable
def test_block_mask_no_overlap():
    db = DropBlock(block_size=2, gamma=0.1)
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
    db = DropBlock(block_size=3, gamma=0.1)
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


if __name__ == '__main__':
    test_block_mask_overlap()
