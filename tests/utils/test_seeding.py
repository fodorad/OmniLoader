import random
import unittest

import numpy as np
import torch

from omniloader.utils.seeding import seed_worker


class TestSeedWorker(unittest.TestCase):
    def _draw_after_seed(self):
        torch.manual_seed(1234)  # sets torch.initial_seed()
        seed_worker(0)
        return np.random.rand(3).tolist(), [random.random() for _ in range(3)]

    def test_deterministic_given_same_torch_seed(self):
        a = self._draw_after_seed()
        b = self._draw_after_seed()
        self.assertEqual(a, b)

    def test_differs_with_different_torch_seed(self):
        torch.manual_seed(1)
        seed_worker(0)
        first = np.random.rand(3).tolist()
        torch.manual_seed(2)
        seed_worker(0)
        second = np.random.rand(3).tolist()
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
