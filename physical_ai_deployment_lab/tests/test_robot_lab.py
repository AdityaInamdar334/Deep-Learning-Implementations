import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robot_lab import collect_demonstrations,fit_policy,rollout,expert_action,compare
class TestLab(unittest.TestCase):
    def test_data(self):
        X,Y=collect_demonstrations(50); self.assertEqual(X.shape,(50,3)); self.assertEqual(Y.shape,(50,2))
    def test_fitted(self):
        X,Y=collect_demonstrations(600); W=fit_policy(X,Y)
        self.assertEqual(W.shape,(3,2)); self.assertTrue(np.isfinite(W).all())
    def test_limits(self):
        self.assertTrue((np.abs(expert_action([-10,-10],[10,10]))<=0.35).all())
    def test_feedback_recovers_better(self):
        _,cases=compare(); self.assertLess(cases[1]['final_error'], cases[0]['final_error'])
    def test_repeatability(self):
        a=compare()[1]; b=compare()[1]
        self.assertTrue(np.array_equal(a[1]['path'],b[1]['path']))
if __name__=='__main__': unittest.main()
