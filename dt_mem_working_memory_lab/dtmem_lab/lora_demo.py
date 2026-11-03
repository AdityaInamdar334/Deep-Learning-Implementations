"""LoRA mechanics: adapting a frozen projection with a low-rank update.

This exercise does NOT reproduce the paper's downstream Meta-World LoRA results.
"""
import json
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, frozen_weight, rank=2, alpha=1.0):
        super().__init__()
        out_features, in_features = frozen_weight.shape
        self.weight = nn.Parameter(frozen_weight.detach().clone(), requires_grad=False)
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return x @ self.weight.T + self.scale * (x @ self.A.T) @ self.B.T


def run(seed=11, steps=350, rank=2):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    # The unknown target differs from the frozen matrix by a rank-2 update.
    base = torch.randn(8, 12) / 3
    true_A = torch.randn(2, 12) / 3
    true_B = torch.randn(8, 2) / 3
    target = base + true_B @ true_A
    x = torch.randn(320, 12)
    y = x @ target.T
    model = LoRALinear(base, rank=rank, alpha=rank)
    before = float(((model(x) - y)**2).mean().detach())
    opt = torch.optim.Adam([model.A, model.B], lr=0.04)
    for _ in range(steps):
        loss = ((model(x) - y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    after = float(((model(x) - y)**2).mean().detach())
    return {'initial_mse': before, 'final_mse': after,
            'rank': rank, 'base_parameters_frozen': int(model.weight.numel()),
            'trainable_adapter_parameters': int(model.A.numel() + model.B.numel()),
            'total_linear_weight_parameters': int(model.weight.numel())}


if __name__=='__main__':
    print(json.dumps(run(), indent=2))
