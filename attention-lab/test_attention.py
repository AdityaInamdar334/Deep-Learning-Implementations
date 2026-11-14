import unittest
from attention import dense, scores, tiled


class AttentionTests(unittest.TestCase):
    def test_geometry(self):
        self.assertEqual(scores(64, 64, 4096), 256)
        self.assertEqual(scores(64, 64, 77), 4.8125)

    def test_resolution_growth(self):
        self.assertEqual(scores(128, 128, 16384) / scores(64, 64, 4096), 16)
        self.assertEqual(scores(128, 128, 77) / scores(64, 64, 77), 4)

    def test_tiling_with_new_maximum(self):
        logits = [1, 2, 1001, 1000, 999]
        values = [-2, 4, 7, 0, -1]
        for block in (1, 2, 3, 5, 8):
            self.assertAlmostEqual(tiled(logits, values, block), dense(logits, values))

    def test_constant_values(self):
        self.assertAlmostEqual(tiled([8, -3, 5], [2, 2, 2], 2), 2)
