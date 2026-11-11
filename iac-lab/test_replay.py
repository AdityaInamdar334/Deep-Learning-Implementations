import unittest
from replay import Turn, fit, priority, replay

trace = [Turn(0, 1, "a", 1, 0, 8, 2), Turn(1, 2, "a", 1, 0, 4, 1),
         Turn(2, 1, "a", 2, 10, 1, 2)]


class ReplayTests(unittest.TestCase):
    def test_counters(self):
        window, prob = fit(trace)["a", 1]
        self.assertEqual((window, prob), (2, .5))

    def test_lexicographic_and_boundary(self):
        stats = {("a", 1): (3, .9), ("b", 1): (8, .1)}
        self.assertLess(priority("intent", (0,"a",1,100),3,stats),
                        priority("intent", (0,"b",1,1),3,stats))

    def test_weighting(self):
        self.assertEqual(replay(trace, 100, "lru", {}), (10, 23, 0))

    def test_pressure(self):
        self.assertEqual(replay(trace, 10, "lru", {}), (0, 23, 0))

    def test_no_capacity(self):
        for policy in ("lru", "intent", "turn"):
            self.assertEqual(replay(trace, 0, policy, {}), (0,23,0))
