import unittest
import torch
from hourglass_lab.models import (causal_shift,upsample_repeat,AverageShortener,LinearShortener,
                                  LinearUpsampler,HourglassLM,VanillaLM)
from hourglass_lab.data import repeat_batch

class TestHourglass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def test_shift_zero(self):
        x=torch.arange(5).view(1,5,1);self.assertTrue(torch.equal(causal_shift(x,0),x))
    def test_shift_kminus1(self):
        x=torch.arange(5).view(1,5,1)
        self.assertEqual(causal_shift(x,2).flatten().tolist(),[0,0,0,1,2])
    def test_shift_longer_than_sequence(self):
        self.assertEqual(causal_shift(torch.ones(1,4,2),6).sum().item(),0)
    def test_invalid_shift(self):
        with self.assertRaises(ValueError):causal_shift(torch.ones(1,4,2),-1)
    def test_avg_pool(self):
        x=torch.arange(1,6,dtype=torch.float32).view(1,5,1)
        self.assertTrue(torch.allclose(AverageShortener()(x,2).flatten(),torch.tensor([1.5,3.5,5.])))
    def test_repeat_truncate(self):
        x=torch.tensor([[[3.],[5.],[7.]]]);self.assertEqual(upsample_repeat(x,2,5).flatten().tolist(),[3,3,5,5,7])
    def test_linear_shapes_and_padding(self):
        x=torch.randn(2,13,8)
        self.assertEqual(LinearShortener(8,3)(x,3).shape,(2,5,8))
        self.assertEqual(LinearUpsampler(8,3)(torch.randn(2,5,8),13,3).shape,(2,13,8))
    def test_unmatched_linear_factor(self):
        with self.assertRaises(ValueError): LinearShortener(8,3)(torch.randn(2,6,8),2)
    def test_vanilla_output_shape(self):
        self.assertEqual(VanillaLM(16,16,4,4)(torch.randint(16,(2,17))).shape,(2,17,16))
    def test_hourglass_nondivisible(self):
        x=torch.randint(16,(2,25))
        for k in [2,3,4]:
            self.assertEqual(HourglassLM(16,16,4,factors=(k,))(x).shape,(2,25,16))
    def test_nested_hourglass(self):
        self.assertEqual(HourglassLM(16,16,4,factors=(2,2))(torch.randint(16,(2,27))).shape,(2,27,16))
    def test_variable_shortening(self):
        m=HourglassLM(16,16,4,factors=(3,))
        x=torch.randint(16,(2,23))
        self.assertEqual(m(x,override_factor=2).shape,(2,23,16))
    def test_model_gradients(self):
        m=HourglassLM(16,16,4)
        m(torch.randint(16,(2,17))).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()))
    def test_causality_vanilla(self):
        self.causality(VanillaLM(16,16,4,layers=3))
    def test_causality_k2(self):
        self.causality(HourglassLM(16,16,4,factors=(2,)))
    def test_causality_k3(self):
        self.causality(HourglassLM(16,16,4,factors=(3,)))
    def test_causality_k4_nondivisible(self):
        self.causality(HourglassLM(16,16,4,factors=(4,)))
    def test_causality_nested(self):
        self.causality(HourglassLM(16,16,4,factors=(2,3)))
    def test_causality_linear(self):
        self.causality(HourglassLM(16,16,4,factors=(3,),pooling='linear'))
    def test_no_shift_leaks(self):
        torch.manual_seed(999)
        m=HourglassLM(16,16,4,factors=(3,),do_shift=False)
        inp=torch.randint(16,(1,12));other=inp.clone();other[:,2]=(other[:,2]+1)%16
        with torch.no_grad():
            delta=(m(inp)[:,:2]-m(other)[:,:2]).abs().max().item()
        self.assertGreater(delta,1e-6)
    def test_shift_fixes_pool_leak(self):
        x=torch.arange(1,13,dtype=torch.float32).view(1,12,1)
        changed=x.clone();changed[:,2]=999
        a=upsample_repeat(AverageShortener()(causal_shift(x,2),3),3,12)
        b=upsample_repeat(AverageShortener()(causal_shift(changed,2),3),3,12)
        self.assertTrue(torch.equal(a[:,:2],b[:,:2]))
    def test_unshifted_pool_leaks(self):
        x=torch.arange(1,13,dtype=torch.float32).view(1,12,1)
        changed=x.clone();changed[:,2]=999
        a=upsample_repeat(AverageShortener()(x,3),3,12)
        b=upsample_repeat(AverageShortener()(changed,3),3,12)
        self.assertFalse(torch.equal(a[:,:2],b[:,:2]))
    def test_synthetic_data_reproducible(self):
        self.assertTrue(torch.equal(repeat_batch(10,25,seed=123),repeat_batch(10,25,seed=123)))
    def test_synthetic_data_seeds_differ(self):
        self.assertFalse(torch.equal(repeat_batch(10,25,seed=12),repeat_batch(10,25,seed=13)))
    def test_prediction_not_equal_future_input(self):
        m=HourglassLM(16,16,4)
        inp=torch.randint(16,(1,17))
        with torch.no_grad():
            a=m(inp);b=m(inp.clone())
        self.assertTrue(torch.equal(a,b))
    def causality(self,m):
        torch.manual_seed(124)
        inp=torch.randint(16,(1,25))
        m.eval()
        with torch.no_grad():
            a=m(inp)
            for j in [1,5,11,19,24]:
                altered=inp.clone();altered[0,j]=(altered[0,j]+3)%16
                b=m(altered)
                delta=(a[:,:j]-b[:,:j]).abs().max().item()
                self.assertLess(delta,2e-5,msg=f'future changed earlier output j={j} delta={delta}')
if __name__=='__main__':unittest.main()
