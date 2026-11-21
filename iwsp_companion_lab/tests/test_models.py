import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from handwritten_models import XorNet, GCDNet, GCDStep

class ModelTests(unittest.TestCase):
    def test_xor_truth_table(self):
        x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        self.assertTrue(torch.equal(XorNet()(x),torch.tensor([0.,1.,1.,0.])))
    def test_no_trainable_parameters(self):
        self.assertEqual(sum(p.numel() for p in XorNet().parameters() if p.requires_grad),0)
    def test_gcd(self):
        pairs=torch.tensor([[84.,36.],[7.,11.],[12.,12.],[30.,1.]])
        expected=torch.tensor([12.,1.,12.,1.])
        self.assertTrue(torch.equal(GCDNet(100)(pairs),expected))
    def test_step(self):
        self.assertTrue(torch.equal(GCDStep()(torch.tensor([[84.,36.]])), torch.tensor([[48.,36.]])))

if __name__=='__main__': unittest.main()
