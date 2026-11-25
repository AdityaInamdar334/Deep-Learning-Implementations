import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from operator_lab import GRID, DeepONet, forcing, make_dataset, solve_trajectory, vector_field

class TestLab(unittest.TestCase):
    def test_vector_field_matches_first_order_equation(self):
        state=np.array([0.7,-0.2]); got=vector_field(0.,state,mu=1.)
        self.assertAlmostEqual(got[0],state[1])
        self.assertAlmostEqual(got[1],(1-state[0]**2)*state[1]-state[0])
    def test_equilibrium(self):
        np.testing.assert_allclose(vector_field(0.,[0.,0.]),[0.,0.])
    def test_initial_conditions(self):
        z=solve_trajectory([0.2,0.3,0.1,0.0],(0.4,-0.5))
        np.testing.assert_allclose(z[0],[0.4,-0.5],atol=1e-12)
    def test_dataset_dimensions(self):
        x,y=make_dataset(3,42)
        self.assertEqual(tuple(x.shape),(3,len(GRID)+2))
        self.assertEqual(tuple(y.shape),(3,len(GRID)))
    def test_model_output_and_gradient(self):
        model=DeepONet(width=16,latent=16)
        xx=torch.randn(2,len(GRID)+2); t=torch.linspace(0,1,13)[:,None]
        output=model(xx,t)
        self.assertEqual(tuple(output.shape),(2,13))
        output.square().mean().backward()
        self.assertIsNotNone(model.branch[0].weight.grad)
    def test_holdout_seed_differs(self):
        a,_=make_dataset(1,42); b,_=make_dataset(1,1042)
        self.assertFalse(torch.allclose(a,b))

if __name__=='__main__': unittest.main()
