import unittest
import torch
import torch.nn as nn
from torch.nn import functional as F
from lora import LoRAProjection, LoRAEmbedding


class TestLoRASuite(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_initialization_invariant(self):
        """B=0 ensures the adapter initially produces 0 output delta."""
        base = nn.Linear(16, 12)
        layer = LoRAProjection(base, rank=4, alpha=8.0)
        x = torch.randn(2, 5, 16)
        layer.eval()
        torch.testing.assert_close(layer(x), base(x))

    def test_base_frozen_and_gradient_flow(self):
        """Base weight must have requires_grad=False; B receives gradients while A does not at step 0."""
        base = nn.Linear(16, 12)
        layer = LoRAProjection(base, rank=4, alpha=8.0)
        x = torch.randn(2, 5, 16)
        layer.train()
        loss = layer(x).square().mean()
        loss.backward()

        self.assertFalse(base.weight.requires_grad)
        self.assertIsNone(base.weight.grad)
        self.assertIsNotNone(layer.A.grad)
        self.assertEqual(torch.count_nonzero(layer.A.grad).item(), 0)
        self.assertGreater(layer.B.grad.abs().sum().item(), 0)

    def test_optimizer_step_enables_a_gradient(self):
        """After B updates to non-zero, A receives non-zero gradients on subsequent backward passes."""
        base = nn.Linear(16, 12)
        layer = LoRAProjection(base, rank=4, alpha=8.0)
        x = torch.randn(2, 5, 16)
        orig_base_weight = base.weight.detach().clone()

        opt = torch.optim.AdamW([layer.A, layer.B], lr=0.01)
        layer.train()
        layer(x).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        layer(x).square().mean().backward()
        self.assertGreater(layer.A.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(orig_base_weight, base.weight))

    def test_merge_copy_exactness(self):
        """Merged weight forward pass must match unmerged forward pass within tight numerical tolerance."""
        base = nn.Linear(16, 12)
        layer = LoRAProjection(base, rank=4, alpha=8.0)
        with torch.no_grad():
            layer.A.normal_(std=0.02)
            layer.B.normal_(std=0.02)
        layer.eval()
        merged = layer.merged_copy()
        x = torch.randn(4, 8, 16)
        torch.testing.assert_close(layer(x), merged(x), rtol=1e-5, atol=1e-6)

    def test_merge_fails_in_training_mode(self):
        """merged_copy() must enforce eval() mode to avoid baking in active dropouts."""
        base = nn.Linear(16, 12)
        layer = LoRAProjection(base, rank=4, alpha=8.0, dropout=0.1)
        layer.train()
        with self.assertRaises(RuntimeError):
            layer.merged_copy()

    def test_invalid_rank(self):
        """Rank must be >= 1."""
        base = nn.Linear(16, 12)
        with self.assertRaises(ValueError):
            LoRAProjection(base, rank=0)
        with self.assertRaises(ValueError):
            LoRAProjection(base, rank=-2)

    def test_transposed_gpt2_weight_layout(self):
        """LoRA correctly adapts GPT-2 Conv1D input-first weight matrix layout."""
        class InputFirstProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(16, 12))
                self.bias = nn.Parameter(torch.randn(12))
            def forward(self, x):
                return x @ self.weight + self.bias

        base = InputFirstProjection()
        layer = LoRAProjection(base, rank=4, alpha=8.0, transposed=True)
        with torch.no_grad():
            layer.B.normal_(std=0.02)
        layer.eval()
        merged = layer.merged_copy()
        x = torch.randn(3, 7, 16)
        torch.testing.assert_close(layer(x), merged(x), rtol=1e-5, atol=1e-6)

    def test_lora_embedding(self):
        """LoRAEmbedding adapts embedding weights while keeping base embedding frozen."""
        base_emb = nn.Embedding(25, 16)
        layer = LoRAEmbedding(base_emb, rank=4, alpha=8.0)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        torch.testing.assert_close(layer(ids), base_emb(ids))

        loss = layer(ids).square().mean()
        loss.backward()
        self.assertIsNone(base_emb.weight.grad)
        self.assertGreater(layer.up.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
