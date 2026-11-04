import torch
from torch import nn
from lora import LoRAProjection

torch.manual_seed(7)
base = nn.Linear(12, 9)
layer = LoRAProjection(base, rank=3, alpha=6)
x = torch.randn(4, 5, 12)
original = base.weight.detach().clone()

layer.eval()
torch.testing.assert_close(layer(x), base(x))
layer.train()
layer(x).square().mean().backward()
assert base.weight.grad is None
assert layer.A.grad is not None
assert torch.count_nonzero(layer.A.grad).item() == 0
assert layer.B.grad.abs().sum().item() > 0

optimizer = torch.optim.AdamW([layer.A, layer.B], lr=0.01,
                              weight_decay=0.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
layer(x).square().mean().backward()
assert layer.A.grad.abs().sum().item() > 0
assert torch.equal(original, base.weight)

layer.eval()
merged = layer.merged_copy()
torch.testing.assert_close(layer(x), merged(x), rtol=1e-5, atol=1e-6)
print('Initialization, gradients, freezing, and merge checks passed.')

# Check GPT-2-style input-first weight storage without downloading a model.
class InputFirstProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(12, 9))
        self.bias = nn.Parameter(torch.randn(9))
    def forward(self, x):
        return x @ self.weight + self.bias

layer = LoRAProjection(InputFirstProjection(), rank=3, transposed=True)
with torch.no_grad():
    layer.B.normal_(std=0.01)
layer.eval()
torch.testing.assert_close(layer(x), layer.merged_copy()(x),
                           rtol=1e-5, atol=1e-6)

from lora import LoRAEmbedding
embedding = LoRAEmbedding(nn.Embedding(20, 12), rank=3)
ids = torch.tensor([[1, 2, 3], [3, 4, 5]])
torch.testing.assert_close(embedding(ids), embedding.base(ids))
embedding(ids).square().mean().backward()
assert embedding.base.weight.grad is None
assert embedding.up.grad.abs().sum().item() > 0
print('Transposed projection and embedding checks passed.')
