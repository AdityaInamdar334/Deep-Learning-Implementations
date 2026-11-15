import unittest
import numpy as np
from lab import assign, distances


class Routing(unittest.TestCase):
    def setUp(self):
        self.c = np.array([[0.], [4.], [5.], [9.]])
        self.index = np.array([[2.], [7.]]), [np.array([0, 1]), np.array([2, 3])]

    def test_missed_winner(self):
        c = np.array([[0.], [4.], [5.], [15.]])
        index = np.array([[2.], [10.]]), self.index[1]
        ids, _ = assign(np.array([[5.1]]), c, index, 1)
        self.assertEqual(ids[0], 1)
        self.assertEqual(distances(np.array([[5.1]]), c).argmin(), 2)

    def test_full_probe(self):
        x = np.array([[.5], [5.1], [9.]])
        ids, work = assign(x, self.c, self.index, 2)
        np.testing.assert_array_equal(ids, distances(x, self.c).argmin(1))
        self.assertEqual(work, 6)
