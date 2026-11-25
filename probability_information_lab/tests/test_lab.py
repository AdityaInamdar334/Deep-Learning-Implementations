import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lab import sigmoid,binary_entropy,kl,posterior_positive,nll,brier,make_calibration_data,fit_temperature

class ProbabilityLabTests(unittest.TestCase):
    def test_bayes(self): self.assertAlmostEqual(posterior_positive(),.08755760368663594,places=8)
    def test_entropy(self):
        self.assertAlmostEqual(binary_entropy(np.array([.5]))[0],1.0)
        self.assertEqual(float(binary_entropy(np.array([0.,1.])).sum()),0.0)
    def test_kl(self):
        p=np.array([.2,.3,.5]);q=np.array([.5,.3,.2]);self.assertGreater(kl(p,q),0);self.assertAlmostEqual(kl(p,p),0)
        self.assertEqual(kl(np.array([1.,0.]),np.array([0.,1.])),np.inf)
    def test_softmax_equiv_logistic(self):
        z=np.array([-2.,0.,2.]);self.assertTrue(np.allclose(sigmoid(z),1/(1+np.exp(-z))))
    def test_temperature_fit(self):
        z,y=make_calibration_data();t=fit_temperature(z[:6000],y[:6000]);self.assertGreater(t,1.5)
        self.assertLess(nll(z[6000:],y[6000:],t),nll(z[6000:],y[6000:]))
    def test_temperature_keeps_decisions(self):
        z,y=make_calibration_data();t=fit_temperature(z[:6000],y[:6000]);self.assertTrue(np.array_equal(sigmoid(z[6000:])>=.5,sigmoid(z[6000:]/t)>=.5))
    def test_brier(self):self.assertAlmostEqual(brier([0.,1.],[0,1]),0.)

if __name__=='__main__':unittest.main()
