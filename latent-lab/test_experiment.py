import unittest
from experiment import SCALE, decode, encode, experiment, mix, predict_noise, schedule


class Contracts(unittest.TestCase):
    def test_round_trip_can_hide_both_conventions(self):
        for factor in (SCALE, 1):
            self.assertAlmostEqual(decode(encode(12.3, factor), factor), 12.3)

    def test_schedule(self):
        a = schedule()
        self.assertEqual(len(a), 1000)
        self.assertAlmostEqual(a[0], 1 - 0.00085)
        self.assertAlmostEqual(a[-1] / a[-2], 1 - 0.012)
        self.assertTrue(all(x > y > 0 for x, y in zip(a, a[1:])))

    def test_oracle_limits(self):
        self.assertEqual(predict_noise(mix(4, 3, 0), 0), 3)
        self.assertEqual(predict_noise(mix(4, 3, 1), 1), 0)

    def test_gaussian_risks(self):
        a = schedule()[500]
        variance = SCALE ** -2
        expected = [a, a*a + (1-a)*a*variance, a*variance/(a*variance+1-a)]
        for measured, theory in zip(experiment(count=60000), expected):
            self.assertLess(abs(measured / theory - 1), 0.03)
