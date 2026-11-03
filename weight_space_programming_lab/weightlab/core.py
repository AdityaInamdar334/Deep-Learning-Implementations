"""Tiny networks with known circuits for controlled read / write experiments."""
from __future__ import annotations
import copy
import torch
from torch import nn

class TinyPolicy(nn.Module):
    """4 inputs -> 16 ReLU units -> 3 actions; action 2 is unproductive 'stall'."""
    def __init__(self, inputs=4, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(inputs, hidden, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 3, bias=True)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.relu(self.fc1(x)))


def make_policy(axis=0, nuisance=3, broken=False):
    if axis == nuisance or axis not in range(4) or nuisance not in range(4):
        raise ValueError("axis and nuisance must be distinct input coordinates")
    m = TinyPolicy()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.fc1.weight[0, axis] = 1
        m.fc1.weight[1, axis] = -1
        m.fc1.weight[2, nuisance] = 1
        m.fc1.weight[3, nuisance] = -1
        m.fc2.weight[0, 0] = 2
        m.fc2.weight[1, 1] = 2
        if broken:
            m.fc2.weight[2, 2] = 4
            m.fc2.weight[2, 3] = 4
    return m


def permute_hidden(model: TinyPolicy, order: torch.Tensor) -> TinyPolicy:
    if sorted(order.tolist()) != list(range(model.fc1.out_features)):
        raise ValueError("order must be a permutation")
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.fc1.weight.copy_(model.fc1.weight[order])
        m.fc1.bias.copy_(model.fc1.bias[order])
        m.fc2.weight.copy_(model.fc2.weight[:, order])
    return m


def rescale_hidden(model: TinyPolicy, unit: int, scale: float) -> TinyPolicy:
    """Positive scaling is an exact ReLU functional symmetry."""
    if scale <= 0: raise ValueError("positive scales only")
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.fc1.weight[unit].mul_(scale)
        m.fc1.bias[unit].mul_(scale)
        m.fc2.weight[:,unit].div_(scale)
    return m


class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def compile_xor() -> XORNet:
    """Write six scalars into an XOR circuit with no optimizer, samples or gradients."""
    m = XORNet()
    with torch.no_grad():
        m.fc1.weight.copy_(torch.tensor([[1.,-1.],[-1.,1.]]))
        m.fc2.weight.copy_(torch.tensor([[1.,1.]]))
    return m


def inspect_axis_naive(model: TinyPolicy) -> int:
    """Bad diagnostic: assumes first hidden unit has a stable identity."""
    return int(model.fc1.weight[0].abs().argmax().item())


def inspect_axis_invariant(model: TinyPolicy) -> int:
    """Better diagnostic for this toy: sums first-layer absolute weights by input."""
    return int(model.fc1.weight.abs().sum(dim=0).argmax().item())
