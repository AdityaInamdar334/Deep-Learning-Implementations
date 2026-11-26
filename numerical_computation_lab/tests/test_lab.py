import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from lab import stable_softmax,stable_log_softmax,stable_logsumexp,gradient_descent,quadratic_loss,least_squares_on_ball
class LabTests(unittest.TestCase):
    def test_extreme_softmax(self):
        p=stable_softmax([1000,999,998]);self.assertTrue(np.isfinite(p).all());self.assertAlmostEqual(p.sum(),1)
    def test_log_softmax_tail(self):
        p=stable_log_softmax([0,-1000]);self.assertAlmostEqual(p[1],-1000,places=5)
    def test_shift_invariance(self):
        self.assertTrue(np.allclose(stable_softmax([2,3]),stable_softmax([1002,1003])))
    def test_logsumexp(self):
        self.assertAlmostEqual(float(stable_logsumexp(np.array([1000.,1000.]))),1000+np.log(2))
    def test_gd_decreases(self):
        H=np.diag([80.,1.]);p=gradient_descent([3.,3.],H,.005,30);self.assertLess(quadratic_loss(p[-1],H),quadratic_loss(p[0],H))
    def test_newton_quadratic(self):
        H=np.diag([80.,1.]);x=np.array([3.,3.]);self.assertTrue(np.allclose(x-np.linalg.solve(H,H@x),np.zeros(2)))
    def test_ball_constraint_active(self):
        A=np.eye(2);b=np.array([2.,1.]);x,lam=least_squares_on_ball(A,b,1.);self.assertAlmostEqual(np.linalg.norm(x),1.,places=8);self.assertGreater(lam,0);self.assertTrue(np.linalg.norm(A.T@(A@x-b)+2*lam*x)<1e-8)
    def test_ball_constraint_inactive(self):
        A=np.eye(2);b=np.array([.2,.3]);x,lam=least_squares_on_ball(A,b,1.);self.assertAlmostEqual(lam,0);self.assertTrue(np.allclose(x,b))
if __name__=='__main__':unittest.main()
