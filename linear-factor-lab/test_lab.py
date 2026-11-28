import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import ppca,posterior,sfa,ista,recovery
class Tests(unittest.TestCase):
    def test_ppca(self):
        x=np.random.default_rng(1).normal(size=(200,4)); m,w,s=ppca(x,2); h,c=posterior(x,m,w,s); self.assertEqual(h.shape,(200,2)); self.assertTrue(np.all(np.linalg.eigvalsh(c)>0)); self.assertGreater(s,0)
    def test_sfa(self):
        t=np.arange(2000); x=np.c_[np.sin(t*.01),np.sin(t*.8)]; m,w,v=sfa(x,1); self.assertGreater(abs(np.corrcoef(((x-m)@w)[:,0],x[:,0])[0,1]),.99)
    def test_ista(self):
        x=np.array([[2.,-.2,-3.]]); np.testing.assert_allclose(ista(x,np.eye(3),.5),[[1.5,0,-2.5]])
    def test_recovery(self):
        x=np.random.default_rng(2).normal(size=(100,2)); self.assertAlmostEqual(recovery(x,-x[:,::-1]),1)
if __name__=='__main__': unittest.main()
