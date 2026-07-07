import unittest

from omniloader.sampling.subsamplers import ExhaustionPolicy, IndexPool


class TestIndexPool(unittest.TestCase):
    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            IndexPool(0)

    def test_zero_count_returns_empty(self):
        self.assertEqual(IndexPool(10).draw(0, epoch=0), [])

    def test_fresh_without_replacement_no_repeats(self):
        pool = IndexPool(10, replacement=False, policy=ExhaustionPolicy.FRESH, seed=1)
        drawn = pool.draw(10, epoch=0)
        self.assertEqual(sorted(drawn), list(range(10)))

    def test_fresh_is_deterministic_per_epoch(self):
        a = IndexPool(20, seed=3).draw(5, epoch=2)
        b = IndexPool(20, seed=3).draw(5, epoch=2)
        self.assertEqual(a, b)

    def test_fresh_differs_across_epochs(self):
        pool = IndexPool(50, seed=3)
        self.assertNotEqual(pool.draw(10, epoch=0), pool.draw(10, epoch=1))

    def test_fresh_oversample_tiles(self):
        pool = IndexPool(4, policy=ExhaustionPolicy.FRESH, seed=0)
        drawn = pool.draw(10, epoch=0)
        self.assertEqual(len(drawn), 10)
        self.assertTrue(all(0 <= i < 4 for i in drawn))

    def test_replacement_allows_repeats(self):
        pool = IndexPool(3, replacement=True, seed=0)
        drawn = pool.draw(30, epoch=0)
        self.assertEqual(len(drawn), 30)
        self.assertTrue(all(0 <= i < 3 for i in drawn))

    def test_exhaust_covers_before_reuse(self):
        pool = IndexPool(5, policy=ExhaustionPolicy.EXHAUST, seed=0)
        first = pool.draw(3, epoch=0)
        second = pool.draw(2, epoch=1)
        # First full pass (5 draws) must cover every index exactly once.
        self.assertEqual(sorted(first + second), list(range(5)))

    def test_exhaust_wraps_across_reshuffles(self):
        pool = IndexPool(4, policy=ExhaustionPolicy.EXHAUST, seed=0)
        drawn = pool.draw(10, epoch=0)
        self.assertEqual(len(drawn), 10)
        # Each index appears at least twice within 10 draws over size 4.
        for i in range(4):
            self.assertGreaterEqual(drawn.count(i), 2)

    def test_weighted_draw_favours_high_weight(self):
        # Only index 2 has weight -> every draw is index 2.
        pool = IndexPool(4, seed=0, weights=[0.0, 0.0, 1.0, 0.0])
        drawn = pool.draw(5, epoch=0)  # count > 1 nonzero -> replacement kicks in
        self.assertEqual(drawn, [2, 2, 2, 2, 2])

    def test_weighted_without_replacement(self):
        pool = IndexPool(4, seed=0, weights=[1.0, 1.0, 1.0, 1.0])
        drawn = pool.draw(4, epoch=0)
        self.assertEqual(sorted(drawn), [0, 1, 2, 3])

    def test_weights_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            IndexPool(4, weights=[1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
