import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from representation_lab import make_rings, radial_features, fit_logistic, predict

class RepresentationTests(unittest.TestCase):
    def test_reproducible(self):
        a,b=make_rings(30, 7); c,d=make_rings(30,7)
        self.assertTrue(np.array_equal(a,c));self.assertTrue(np.array_equal(b,d))
    def test_shapes(self):
        x,y=make_rings(50)
        self.assertEqual(x.shape,(50,2));self.assertEqual(y.shape,(50,))
    def test_radial_features(self):
        self.assertTrue(np.allclose(radial_features(np.array([[3.,4.],[0.,2.]])),[[25.],[4.]]))
    def test_accuracy(self):
        x,y=make_rings(400, 8)
        a,b=make_rings(200,9)
        m=fit_logistic(radial_features(x),y,steps=1200)
        self.assertGreater(np.mean(predict(m,radial_features(a))==b),0.97)
    def test_probabilities(self):
        from representation_lab import sigmoid
        self.assertTrue(np.all((sigmoid(np.array([-1000,0,1000])) >= 0)&(sigmoid(np.array([-1000,0,1000]))<=1)))

if __name__=='__main__': unittest.main()
