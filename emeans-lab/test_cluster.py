import unittest
import numpy as np
from cluster import assign, means, revive, stats, train


class ClusterTests(unittest.TestCase):
    def test_assign_matches_direct_distance(self):
        rng = np.random.default_rng(33)
        x, c = rng.normal(size=(40, 5)), rng.normal(size=(7, 5))
        direct = ((x[:, None] - c)**2).sum(axis=2).argmin(axis=1)
        np.testing.assert_array_equal(assign(x, c), direct)

    def test_counts_weight_updates_and_keep_missing_center(self):
        x = np.array([[2.], [8.], [8.], [8.]])
        s, n = stats(x, np.array([0, 0, 0, 0]), 2)
        np.testing.assert_allclose(means(s, n, np.array([[0.], [4.]])), [[6.5], [4.]])

    def test_revival_splits_donor(self):
        sums = np.array([[6., 12.], [0., 0.]])
        counts = np.array([6., 0.])
        splits = revive(sums, counts, np.random.default_rng(1))
        self.assertEqual(splits, 1)
        np.testing.assert_allclose(counts, [3, 3])
        np.testing.assert_allclose(sums.sum(axis=0), [6., 12.])

    def test_lloyd_training_error_never_increases(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(100, 3))
        loss = []
        for _, c, _, _ in train(x, x[:5], 'lloyd', 12, 16, 3):
            loss.append(np.square(x - c[assign(x, c)]).sum())
        self.assertTrue(np.all(np.diff(loss) <= 1e-9))

    def test_one_cluster_final_mean_includes_short_batch(self):
        x = np.arange(14.).reshape(-1, 1)
        for _, _, final, _ in train(x, x[:1], 'ema', 3, 4, 2):
            np.testing.assert_allclose(final[0], x.mean(axis=0))


if __name__ == '__main__':
    unittest.main()
