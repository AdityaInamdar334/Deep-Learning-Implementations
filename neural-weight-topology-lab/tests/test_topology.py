import unittest
import numpy as np
from weight_topology_lab.topology import (pairwise_euclidean, normalize_rows, rips_h0_h1, summarize, betti, weight_cloud)
from weight_topology_lab.experiment import (gaussian_data, permuted_labels, label_for_epoch, label_digest, build_mlp)

class PersistenceTests(unittest.TestCase):
    def test_zero_dim(self):
        self.assertEqual(rips_h0_h1([[0,0], [1,0], [0,1]])[0].shape[0],2)
    def test_square_loop(self):
        # Square: edges at 1, diagonals at sqrt(2), so one persistent loop
        d=rips_h0_h1(np.array([[0,0],[1,0],[1,1],[0,1]]))[1]
        self.assertEqual(len(d),1)
        self.assertAlmostEqual(d[0,0],1.,places=8)
        self.assertAlmostEqual(d[0,1],np.sqrt(2),places=8)
    def test_triangle_no_h1(self):
        self.assertEqual(len(rips_h0_h1([[0,0],[1,0],[0.5,0.866]])[1]),0)
    def test_invariant_under_row_permutation(self):
        x=np.random.default_rng(0).normal(size=(10,3))
        a=rips_h0_h1(x); b=rips_h0_h1(x[::-1])
        for k in [0,1]:
            self.assertTrue(np.allclose(np.sort(a[k],axis=0),np.sort(b[k],axis=0)))
    def test_invariant_under_rotation(self):
        rng=np.random.default_rng(8); x=rng.normal(size=(8,2))
        theta=.67; rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        a=rips_h0_h1(x);b=rips_h0_h1(x@rot)
        for k in (0,1):
            self.assertAlmostEqual(summarize(a[k])['total'],summarize(b[k])['total'],places=6)
    def test_scaling_changes_raw(self):
        x=np.array([[0,0],[1,0],[1,1],[0,1]])
        a=summarize(rips_h0_h1(x)[1])['total'];b=summarize(rips_h0_h1(2*x)[1])['total']
        self.assertAlmostEqual(b,2*a)
    def test_normalization_unit_length(self):
        x=np.array([[3,4],[5,12.],[0,0]])
        y=normalize_rows(x)
        self.assertTrue(np.allclose(np.linalg.norm(y,axis=1),[1,1,0]))
    def test_pairwise(self):
        d=pairwise_euclidean([[0.,0.],[3.,4.]])
        self.assertEqual(d[0,1],5.)
    def test_bad_input(self):
        with self.assertRaises(ValueError): rips_h0_h1(np.zeros((17,3)))
        with self.assertRaises(ValueError): pairwise_euclidean([[float('nan'),1]])
    def test_betti(self):
        d=np.array([[1.,2.],[1.5,3.]])
        self.assertEqual(betti(d,np.array([.5,1.2,1.7,3.] )).tolist(),[0,1,2,0])
    def test_empty(self):
        self.assertEqual(summarize([])['entropy'],0)
    def test_neuron_rows(self):
        w=np.ones((5,10))
        self.assertEqual(weight_cloud(w,count=3).shape,(3,10))
    def test_fixed_labels(self):
        labels=np.tile(np.arange(3),10)
        self.assertTrue(np.array_equal(label_for_epoch(labels,'fixed_random',12,0),label_for_epoch(labels,'fixed_random',12,3)))
    def test_changing_labels(self):
        labels=np.tile(np.arange(3),10)
        self.assertFalse(np.array_equal(label_for_epoch(labels,'changing_random',12,0),label_for_epoch(labels,'changing_random',12,1)))
    def test_histogram_preserved(self):
        labels=np.tile(np.arange(3),10)
        x=permuted_labels(labels,42)
        self.assertEqual(np.bincount(x).tolist(),np.bincount(labels).tolist())
    def test_label_digests(self):
        self.assertNotEqual(label_digest(np.array([1,2])),label_digest(np.array([2,1])))
    def test_data_split(self):
        x,y,tx,ty=gaussian_data(7)
        self.assertEqual(x.shape,(360,2));self.assertEqual(len(ty),180)
    def test_model_shapes(self):
        import torch
        self.assertEqual(build_mlp()(torch.zeros(4,2)).shape,(4,3))

if __name__=='__main__': unittest.main()
