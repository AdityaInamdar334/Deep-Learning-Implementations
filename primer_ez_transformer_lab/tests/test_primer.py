import unittest
import torch
from torch import nn
from primer_lab.data import periodic_sequences, teacher_forcing
from primer_lab.model import (SquaredReLU, CausalDepthwiseConv1d, CausalSelfAttention,
                              make_model, ModelConfig, TinyLanguageModel, num_parameters,
                              ABLATIONS)
from primer_lab.benchmark import run_trial, summarize


class PrimerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_squared_relu_exact(self):
        x = torch.tensor([-3., -1., 0., 2., 3.])
        self.assertTrue(torch.equal(SquaredReLU()(x), torch.tensor([0., 0., 0., 4., 9.])))

    def test_squared_relu_gradient(self):
        x = torch.tensor([-2., 0., 3.], requires_grad=True)
        SquaredReLU()(x).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.tensor([0., 0., 6.])))

    def test_grouped_conv(self):
        c = CausalDepthwiseConv1d(4, 8)
        self.assertEqual(c.conv.groups, 8)
        self.assertEqual(c.conv.weight.shape, (8, 1, 3))

    def test_shared_head_permutation_equivariance(self):
        conv = CausalDepthwiseConv1d(3, 4, shared_across_heads=True)
        x = torch.randn(2, 3, 7, 4)
        perm = [2, 0, 1]
        self.assertTrue(torch.allclose(conv(x)[:, perm], conv(x[:, perm]), atol=1e-6))

    def test_per_head_conv(self):
        c = CausalDepthwiseConv1d(4, 8, shared_across_heads=False)
        self.assertEqual(c.conv.groups, 32)
        self.assertEqual(c.conv.weight.shape, (32, 1, 3))

    def test_conv_causality_shared(self):
        self._check_conv_causality(True)

    def test_conv_causality_per_head(self):
        self._check_conv_causality(False)

    def _check_conv_causality(self, shared):
        c = CausalDepthwiseConv1d(2, 3, 3, shared)
        a = torch.randn(2, 2, 7, 3)
        b = a.clone()
        b[:, :, 4:] += 17
        self.assertTrue(torch.allclose(c(a)[:, :, :4], c(b)[:, :, :4], atol=1e-6))

    def test_conv_no_cross_channel_information(self):
        c = CausalDepthwiseConv1d(2, 3)
        a = torch.zeros(1, 2, 6, 3)
        b = a.clone(); b[:, 0, 1, 0] = 10
        delta = c(b) - c(a)
        self.assertTrue(torch.allclose(delta[:, 0, :, 1:], torch.zeros_like(delta[:, 0, :, 1:])))

    def test_conv_expected_receptive_field(self):
        c = CausalDepthwiseConv1d(1, 1, kernel_size=3)
        with torch.no_grad():
            c.conv.weight.fill_(1.0)
            c.conv.bias.zero_()
        x = torch.tensor([1., 2., 3., 4.]).reshape(1, 1, 4, 1)
        self.assertTrue(torch.equal(c(x).flatten(), torch.tensor([1., 3., 6., 9.])))

    def test_attention_output_shape(self):
        a = CausalSelfAttention(24, 4, use_dconv=True)
        self.assertEqual(a(torch.randn(2, 7, 24)).shape, (2, 7, 24))

    def test_attention_causality(self):
        for use_dconv in (False, True):
            a = CausalSelfAttention(24, 4, use_dconv=use_dconv)
            x = torch.randn(2, 10, 24)
            y = x.clone(); y[:, 5:] += 10
            self.assertTrue(torch.allclose(a(x)[:, :5], a(y)[:, :5], atol=1e-5))

    def test_conv_selective_parameter_count(self):
        full = make_model('dconv_only')
        part = make_model('q_only')
        self.assertLess(num_parameters(part), num_parameters(full))
        self.assertGreater(num_parameters(part), num_parameters(make_model('baseline_relu')))

    def test_invalid_conv_targets(self):
        with self.assertRaises(ValueError):
            CausalSelfAttention(24, 4, use_dconv=True, conv_targets='x')

    def test_model_causality_all_variants(self):
        for name in ABLATIONS:
            m = make_model(name, context=12, d_model=24, heads=4, d_ff=48, layers=1)
            m.eval()
            x = torch.randint(0, 16, (2, 12))
            y = x.clone(); y[:, 6:] = (y[:, 6:] + 2) % 16
            with torch.no_grad():
                a, b = m(x)[:, :6], m(y)[:, :6]
            self.assertTrue(torch.allclose(a, b, atol=1e-5), name)

    def test_context_limit(self):
        m = make_model('primer_ez', context=8)
        with self.assertRaises(ValueError):
            m(torch.zeros(1, 9, dtype=torch.long))

    def test_requires_divisible_heads(self):
        with self.assertRaises(ValueError):
            CausalSelfAttention(26, 4)

    def test_more_parameters_with_depthwise(self):
        a = make_model('baseline_relu')
        b = make_model('primer_ez')
        self.assertGreater(num_parameters(b), num_parameters(a))
        self.assertEqual(num_parameters(make_model('squared_relu_only')), num_parameters(a))

    def test_head_sharing_parameter_difference(self):
        shared = make_model('primer_ez')
        independent = make_model('primer_per_head')
        self.assertGreater(num_parameters(independent), num_parameters(shared))

    def test_data_reproducibility(self):
        a = periodic_sequences(3, 10, 4, 16, torch.Generator().manual_seed(5), 0)
        b = periodic_sequences(3, 10, 4, 16, torch.Generator().manual_seed(5), 0)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, 4:], a[:, :-4]))

    def test_dataset_no_leakage(self):
        a = periodic_sequences(4, 11, 4, 16, torch.Generator().manual_seed(2), .1)
        x, y = teacher_forcing(a)
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertEqual(y.shape, (4, 11))

    def test_data_reject_invalid(self):
        with self.assertRaises(ValueError):
            periodic_sequences(2, 10, 0, 16, torch.Generator())

    def test_trial_smoke(self):
        x = run_trial('primer_ez', 0, steps=2, batch_size=4, context=10,
                      d_model=16, layers=1, d_ff=32, eval_batches=1)
        self.assertGreater(x['training_seconds'], 0)
        self.assertGreater(x['in_distribution']['nll'], 0)
        self.assertEqual(len(summarize([x])), 1)

    def test_malformed_conv_input(self):
        c = CausalDepthwiseConv1d(2, 2)
        with self.assertRaises(ValueError):
            c(torch.randn(2, 5, 2))

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError):
            make_model('not-a-model')

    def test_backward_all_variants(self):
        for name in ABLATIONS:
            m = make_model(name, context=10, d_model=16, heads=4, layers=1, d_ff=32)
            x = torch.randint(0, 16, (3, 10))
            loss = nn.functional.cross_entropy(m(x).reshape(-1, 16), torch.randint(0,16,(30,)))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
