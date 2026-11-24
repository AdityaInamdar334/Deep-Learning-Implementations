import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from oscillator import Oscillator, modes, coefficients, components, numerical_solution

class TestOscillator(unittest.TestCase):
    def setUp(self): self.p=Oscillator()
    def test_poles(self):
        values=modes(self.p)
        self.assertTrue(np.allclose(sorted(values[:2],key=lambda z:z.imag),[-.3- np.sqrt(3.91)*1j,-.3+np.sqrt(3.91)*1j]))
    def test_initial_conditions(self):
        p=Oscillator(x0=.6,v0=-.4)
        t=np.array([0,1e-5])
        val=components(t,p)[2]
        self.assertAlmostEqual(val[0],p.x0,places=10)
        self.assertAlmostEqual((val[1]-val[0])/1e-5,p.v0,places=4)
    def test_exact_vs_numerical(self):
        t=np.linspace(0,30,1501)
        err=np.max(np.abs(components(t,self.p)[2]-numerical_solution(t,self.p)))
        self.assertLess(err,2e-8)
    def test_phase_amplitude(self):
        A,phi,*_=coefficients(self.p)
        self.assertAlmostEqual(A,1/np.hypot(4-1.2**2,.6*1.2))
        self.assertTrue(0<phi<np.pi/2)
    def test_bad_damping(self):
        with self.assertRaises(ValueError):coefficients(Oscillator(c=4))
    def test_components_sum(self):
        t=np.array([0.,1.,2.])
        tr,ss,total=components(t,self.p)
        np.testing.assert_allclose(tr+ss,total)
if __name__=='__main__': unittest.main()
