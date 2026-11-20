import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from moe_lab import create_weights, ExpertStore, LRUExpertCache, routing, forward_one, generate_inputs

class ExpertTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        create_weights(self.root,n_experts=8,hidden=16,intermediate=24)
        import numpy as np
        self.router=torch.from_numpy(np.load(self.root/'router.npy').copy())

    def tearDown(self):self.temp.cleanup()

    def test_cache_matches_no_cache(self):
        xs=generate_inputs(self.router,24,42,'hotspot')
        idx,gates=routing(xs,self.router,top_k=2)
        a=LRUExpertCache(ExpertStore(self.root),0)
        b=LRUExpertCache(ExpertStore(self.root),4)
        for x,i,g in zip(xs,idx,gates):
            torch.testing.assert_close(forward_one(x,i,g,a),forward_one(x,i,g,b),rtol=0,atol=0)
        self.assertGreater(b.hits,0)
        self.assertLessEqual(len(b.entries),4)
        self.assertEqual(a.hits,0)

    def test_lru_eviction(self):
        cache=LRUExpertCache(ExpertStore(self.root),2)
        cache.get(0);cache.get(1);cache.get(0);cache.get(2)
        self.assertEqual(list(cache.entries),[0,2])
        self.assertEqual(cache.hits,1)
        self.assertEqual(cache.misses,3)

    def test_routing_probabilities(self):
        x=torch.randn(5,16)
        ids,gates=routing(x,self.router,2)
        self.assertEqual(ids.shape,(5,2))
        torch.testing.assert_close(gates.sum(dim=-1),torch.ones(5))

if __name__=='__main__':unittest.main()
