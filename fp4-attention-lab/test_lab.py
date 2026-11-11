import unittest
from lab import SCALES, block, nearest, old_token_output, query_experiment


class AttentionChecks(unittest.TestCase):
    def test_scale_grid(self):
        self.assertEqual((len(SCALES), SCALES[1], SCALES[-1]), (127, 2**-9, 448))
        self.assertEqual(nearest(2**-10, SCALES), 0)
        self.assertEqual(nearest(3 * 2**-10, SCALES), 2**-8)

    def test_signed_round_trip(self):
        values = [-6, -1.5, 0, .5, 3, 6]
        self.assertEqual(block(values), (values, 1))

    def test_collective_weight(self):
        self.assertEqual(old_token_output(2048, 1, 6), (0, 0))
        self.assertEqual(old_token_output(2048, 8, 6), (0, 0))
        self.assertEqual(old_token_output(2048, 8, 2), (2**-9, .5))

    def test_repair_has_a_limit(self):
        self.assertEqual(old_token_output(8192, 8, 2), (0, 0))

    def test_query_residual(self):
        for seed in range(12):
            coarse, repaired = query_experiment(seed)
            self.assertLess(repaired, coarse)


if __name__ == "__main__":
    unittest.main()
