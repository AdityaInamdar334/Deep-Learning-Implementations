import unittest
import numpy as np
from lab import schedule, forward, oracle, reverse_step, sample, features, train


class Checks(unittest.TestCase):
    def test_schedule_indexing(self):
        b,a,bar = schedule(1000)
        self.assertEqual(bar[0],a[0])
        self.assertAlmostEqual(bar[-1],.0000403583,9)
        self.assertTrue(np.all(np.diff(bar)<0))

    def test_forward_moments(self):
        noise = np.random.default_rng(5).normal(size=80000)
        x = forward(2.,noise,.36)
        self.assertAlmostEqual(x.mean(),1.2,2)
        self.assertAlmostEqual(x.var(),.64,2)

    def test_last_mean_with_actual_noise(self):
        b,a,bar = schedule(10)
        x = forward(2.,-.3,bar[0])
        self.assertAlmostEqual(reverse_step(x,-.3,b[0],a[0],bar[0],0.),2.)

    def test_oracle_matches_density_score(self):
        a = .55
        m,v = 2*np.sqrt(a), a*.3**2 + 1-a
        def log_density(x):
            return np.logaddexp(-(x-m)**2/(2*v),-(x+m)**2/(2*v))
        x = np.array([-.8,0.,2.1])
        h = 1e-5
        derivative = (log_density(x+h)-log_density(x-h))/(2*h)
        np.testing.assert_allclose(oracle(x,a),-np.sqrt(1-a)*derivative,atol=1e-8)

    def test_training_and_replay(self):
        s = schedule(12)
        w = train(np.random.default_rng(1),s[2],128)
        def draw():
            return sample(np.random.default_rng(2),s,lambda x,i:features(x)@w[i],32)
        np.testing.assert_array_equal(draw(),draw())
        self.assertTrue(np.isfinite(draw()).all())


if __name__ == "__main__":
    unittest.main()
