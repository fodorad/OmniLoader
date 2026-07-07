import unittest

import torch

from omniloader.utils.padding import (
    fill_gaps_with_repeat,
    pad_or_crop_time_dim,
    repeat_pad_time_dim,
)


class TestPadOrCrop(unittest.TestCase):
    def test_pad_matrix(self):
        x = torch.randn(3, 4)
        out, mask = pad_or_crop_time_dim(x, 6)
        self.assertEqual(out.shape, (6, 4))
        self.assertTrue(mask[:3].all())
        self.assertFalse(mask[3:].any())

    def test_crop_matrix(self):
        x = torch.randn(8, 4)
        out, mask = pad_or_crop_time_dim(x, 5)
        self.assertEqual(out.shape, (5, 4))
        self.assertTrue(mask.all())

    def test_exact_match(self):
        x = torch.randn(5, 2)
        out, mask = pad_or_crop_time_dim(x, 5)
        self.assertTrue(torch.equal(out, x))
        self.assertTrue(mask.all())

    def test_vector_pad(self):
        x = torch.randn(3)
        out, mask = pad_or_crop_time_dim(x, 5, pad_value=-1)
        self.assertEqual(out.shape, (5,))
        self.assertTrue(torch.all(out[3:] == -1))
        self.assertFalse(mask[3:].any())


class TestRepeatPad(unittest.TestCase):
    def test_repeats_last_frame(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = repeat_pad_time_dim(x, 4)
        self.assertEqual(out.shape, (4, 2))
        self.assertTrue(torch.equal(out[2], out[3]))
        self.assertTrue(torch.equal(out[3], torch.tensor([3.0, 4.0])))

    def test_no_crop_when_longer(self):
        x = torch.randn(5, 2)
        self.assertTrue(torch.equal(repeat_pad_time_dim(x, 3), x))

    def test_zero_frames_raises(self):
        with self.assertRaises(ValueError):
            repeat_pad_time_dim(torch.zeros(0, 2), 3)


class TestFillGaps(unittest.TestCase):
    def test_fills_forward_and_back(self):
        x = torch.zeros(6, 2)
        x[1] = torch.tensor([1.0, 2.0])
        x[4] = torch.tensor([3.0, 4.0])
        out = fill_gaps_with_repeat(x)
        self.assertTrue(torch.equal(out[0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(out[3], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(out[5], torch.tensor([3.0, 4.0])))

    def test_explicit_mask(self):
        x = torch.ones(3, 2)
        mask = torch.tensor([True, False, True])
        out = fill_gaps_with_repeat(x, mask)
        self.assertEqual(out.shape, (3, 2))

    def test_empty_tensor_returns_same(self):
        x = torch.zeros(0, 2)
        self.assertTrue(torch.equal(fill_gaps_with_repeat(x), x))

    def test_no_valid_frames_raises(self):
        with self.assertRaises(ValueError):
            fill_gaps_with_repeat(torch.zeros(4, 2))


if __name__ == "__main__":
    unittest.main()
