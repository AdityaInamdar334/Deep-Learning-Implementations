import unittest
import torch
import torch.nn.functional as F
from steering_lab.model import ToyResidualPolicy, make_batch, train_model
from steering_lab.interventions import (collect_block_activations, collect_neurons,
    contrast_direction, steer_residual, steer_neuron, project_output_weights, categorical_kl)
from steering_lab.benchmark import run_seed

class TestSteering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_model(0, 20)
        cls.batch, cls.mood, cls.topic = make_batch(32, 100, "test")

    def test_shapes(self):
        m,t = self.model(self.batch)
        self.assertEqual(tuple(m.shape),(32,2)); self.assertEqual(tuple(t.shape),(32,2))
    def test_reproducible_data(self):
        a = make_batch(20,3); b = make_batch(20,3)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
    def test_disjoint_noise_vocab(self):
        a = make_batch(10,3,"train")[0]; b = make_batch(10,3,"test")[0]
        self.assertTrue((a >= 11).any().item() == False)
        self.assertTrue(((b <= 10).sum(-1) == 2).all().item())
    def test_activations_shape(self):
        a = collect_block_activations(self.model,self.batch)
        self.assertEqual(tuple(a.shape), (4,32,48))
    def test_neuron_shape(self):
        n = collect_neurons(self.model, 1, self.batch)
        self.assertEqual(tuple(n.shape),(32,96))
    def test_vector_normalized(self):
        a = self.batch[:16]; b = self.batch[16:]
        d = contrast_direction(self.model,a,b,2)
        self.assertAlmostEqual(d.norm().item(),1,places=5)
    def test_steering_hook_restored(self):
        with torch.no_grad():
            a = self.model(self.batch)[0]
        with steer_residual(self.model,2,torch.ones(48),1.):
            with torch.no_grad():
                b = self.model(self.batch)[0]
        with torch.no_grad():
            c = self.model(self.batch)[0]
        self.assertFalse(torch.equal(a,b)); self.assertTrue(torch.equal(a,c))
    def test_neuron_hook_restored(self):
        with torch.no_grad(): a = self.model(self.batch)[0]
        with steer_neuron(self.model,1,8,10.):
            with torch.no_grad(): b = self.model(self.batch)[0]
        with torch.no_grad(): c = self.model(self.batch)[0]
        self.assertFalse(torch.equal(a,b)); self.assertTrue(torch.equal(a,c))
    def test_projection_copy_no_mutation(self):
        d = F.normalize(torch.randn(48),dim=0)
        orig = self.model.layers[1].mlp.down.weight.clone()
        altered = project_output_weights(self.model,1,d)
        self.assertTrue(torch.equal(orig,self.model.layers[1].mlp.down.weight))
        self.assertLess((d @ altered.layers[1].mlp.down.weight).abs().max().item(),1e-6)
        self.assertLess((d @ altered.layers[1].mlp.down.bias).abs().item(),1e-6)
    def test_kl_properties(self):
        a = torch.randn(12,2)
        self.assertAlmostEqual(categorical_kl(a,a),0.,places=6)
        self.assertGreater(categorical_kl(a,-a),0.)
    def test_benchmark_deterministic(self):
        a = run_seed(5,12,40); b=run_seed(5,12,40)
        self.assertEqual(a,b)

if __name__ == '__main__': unittest.main()
