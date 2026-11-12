import unittest
from collections import Counter
from lab import mcnemar, schedules, toy_decode


class MaskTests(unittest.TestCase):
    def test_matching_by_bin_and_identity(self):
        for seed in range(30):
            masks = schedules(seed)
            for start in range(0, 100, 20):
                counts = [Counter(h for row in mask[start:start+20] for h in row)
                          for mask in masks.values()]
                self.assertEqual(counts[0], counts[1])
                self.assertEqual(counts[0], counts[2])
                self.assertEqual(sum(counts[0].values()), 24)

    def test_spaced_and_burst(self):
        masks = schedules(7)
        for name, stride in [("burst", 1), ("spaced", 3)]:
            active = [i for i, row in enumerate(masks[name][:20]) if row]
            self.assertEqual(active, [active[0]+j*stride for j in range(3)])

    def test_exposure_at_boundaries(self):
        for mask in schedules(3).values():
            self.assertEqual(sum(map(len, mask[:40])), 48)
            self.assertEqual(sum(map(len, mask)), 120)

    def test_copy_state(self):
        damaged = [set(), set(range(8)), set()]
        self.assertEqual(toy_decode(damaged, 13), (0, 2))
        self.assertEqual(toy_decode(damaged, 11), (1, None))

    def test_exact_p(self):
        self.assertEqual(mcnemar(0, 0), 1)
        self.assertAlmostEqual(mcnemar(39, 8), 5.539621596994948e-6)
        self.assertAlmostEqual(mcnemar(36, 18), .019834326728)
