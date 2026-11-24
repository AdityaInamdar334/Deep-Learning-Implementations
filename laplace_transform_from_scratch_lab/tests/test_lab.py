import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from laplace_lab import finite_exponential_integral, exponential_laplace, cosine_laplace, oscillator_response, oscillator

class LaplaceLabTests(unittest.TestCase):
    def test_exponential_converges_in_roc(self):
        s=1+2j;a=.2+.5j
        self.assertAlmostEqual(abs(finite_exponential_integral(s,a,35)-exponential_laplace(s,a)),0,places=8)
    def test_at_pole_finite_integral_grows_with_time(self):
        self.assertAlmostEqual(finite_exponential_integral(1+1j,1+1j,12).real,12)
    def test_cosine_transform(self):
        s=2+.3j
        self.assertAlmostEqual(abs(cosine_laplace(s)-.5*(1/(s-1j)+1/(s+1j))),0,places=12)
    def test_four_oscillator_poles(self):
        r,p,d,_=oscillator()
        self.assertEqual(len(p),4)
        self.assertEqual(len(r),4)
        self.assertEqual(len(d),0)
        self.assertTrue(np.allclose(np.sort_complex(p),np.sort_complex(np.roots(np.polymul([1,.6,4],[1,0,1.2**2])))))
    def test_oscillator_against_numerical_solver(self):
        t=np.linspace(0,12,2401)
        analytical,numerical,_,_=oscillator_response(t)
        self.assertLess(float(np.max(np.abs(analytical-numerical))),5e-4)
        self.assertAlmostEqual(analytical[0],0,places=9)
if __name__=='__main__':unittest.main()
