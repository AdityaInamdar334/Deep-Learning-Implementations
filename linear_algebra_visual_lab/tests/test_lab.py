import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lab import transform_demo,solve_demo,pca_demo
class TestLab(unittest.TestCase):
    def test_eigen_equation(self):
        A,w,V=transform_demo();np.testing.assert_allclose(A@V,V*w,atol=1e-12)
    def test_eigen_orthonormal(self):
        _,_,V=transform_demo();np.testing.assert_allclose(V.T@V,np.eye(2),atol=1e-12)
    def test_condition_number(self):
        self.assertAlmostEqual(solve_demo()['condition_number'],100000.)
    def test_perturbation_amplification(self):
        r=solve_demo();self.assertGreater(r['relative_solution_change'],r['relative_input_change']*100)
    def test_pca_variance(self):
        *_,ratio,mse,S=pca_demo();self.assertAlmostEqual(ratio,float(S[0]**2/np.sum(S**2)))
    def test_pca_reconstruction(self):
        X,R,mean,vec,ratio,mse,S=pca_demo();self.assertLess(mse, np.mean((X-mean)**2));np.testing.assert_allclose((X-R)@vec,np.zeros(len(X)),atol=1e-12)
if __name__=='__main__':unittest.main()
