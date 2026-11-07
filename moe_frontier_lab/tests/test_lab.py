import unittest
import numpy as np
from moe_lab.core import ToyMoE
from moe_lab.planner import estimate, conventional_kv_gb


class MoELabTests(unittest.TestCase):
    def setUp(self):
        self.m = ToyMoE(n_experts=4, hidden=8, intermediate=16, seed=7)
        self.x = np.random.default_rng(3).normal(size=(5, 8)).astype(np.float32)

    def test_all_experts_matches_dense_reference(self):
        np.testing.assert_allclose(self.m.forward(self.x, 4), self.m.dense_reference(self.x), rtol=1e-5, atol=1e-5)

    def test_sparse_routes_and_finite_values(self):
        ids, weights = self.m.routes(self.x, 2)
        self.assertEqual(ids.shape, (5, 2))
        self.assertTrue(np.all(ids >= 0))
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(5), atol=1e-6)
        self.assertTrue(np.isfinite(self.m.forward(self.x, 2)).all())

    def test_stored_weight_count_does_not_change_with_k(self):
        c1, c4 = self.m.cost(1), self.m.cost(4)
        self.assertEqual(c1.total_parameters, c4.total_parameters)
        self.assertLess(c1.active_parameters_per_token, c4.active_parameters_per_token)
        self.assertEqual(c1.weight_bytes_float32, c4.weight_bytes_float32)

    def test_kimi_two_nodes_not_enough_for_fp4_weights(self):
        cap = estimate(2800, 4, gpu_gb=80, gpus_per_node=8)
        self.assertAlmostEqual(cap.raw_weight_gb, 1400)
        self.assertEqual(cap.min_gpus, 18)
        self.assertEqual(cap.min_nodes, 3)

    def test_conventional_kv_doubles_with_tokens(self):
        self.assertAlmostEqual(conventional_kv_gb(32, 8, 128, 2048) * 2,
                               conventional_kv_gb(32, 8, 128, 4096))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            estimate(700, bits=4, gpu_gb=80, reserve_gb_per_gpu=80)
        with self.assertRaises(ValueError):
            self.m.forward(self.x, 0)


if __name__ == '__main__':
    unittest.main()
