import sys, unittest, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import CandidateScorer,decision_loss,probabilities,fit_threshold,selective
from support import generate,policy,next_action
from qwen import ordered,targets
from cost import estimate
class TestLab(unittest.TestCase):
    def test_equivariance(self):
        m=CandidateScorer(); x=torch.randn(12,4,8); p=[2,0,3,1]; torch.testing.assert_close(m(x[:,p]),m(x)[:,p])
    def test_loss(self):
        z=torch.randn(2,4,requires_grad=True); y=torch.eye(4)[:2]; decision_loss(z,y).backward(); self.assertTrue(torch.isfinite(z.grad).all())
    def test_policy(self):
        s=dict(authenticated=False,order_found=True,replacement_eligible=True,wants_replacement=True,urgent=False); self.assertFalse(policy('offer_replacement',s)); self.assertEqual(next_action('returns',s),'authenticate'); self.assertFalse(policy('unknown',s))
    def test_split(self):
        d=generate(); sets=[set(e['case_id'] for e in v) for v in d.values()]; self.assertTrue(all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]))
    def test_targets(self):
        e={'options':[{'id':'a'},{'id':'b'}],'answer':'a','teacher_probs':{'a':.8,'b':.2}}; y=targets([e],[[e['options'][1],e['options'][0]]],'cpu'); torch.testing.assert_close(y[0,:2],torch.tensor([.1,.9])); self.assertAlmostEqual(y.sum().item(),1.)
    def test_gate(self):
        p=np.tile([.6,.4],(50,1)); y=np.ones(50,dtype=int); self.assertEqual(fit_threshold(p,y),1.01); self.assertIsNone(selective(p,y,1.01)['accepted_accuracy'])
    def test_budget(self): self.assertAlmostEqual(estimate(4,.72,800,80,800,160,300)['hybrid_frontier_tokens'],1945.6)
if __name__=='__main__': unittest.main()
