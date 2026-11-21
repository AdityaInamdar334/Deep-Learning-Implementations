"""Educational fixed-weight networks inspired by iWSP, NOT the paper's full reproduction."""
import torch
from torch import nn

class XorNet(nn.Module):
    """Two ReLU detectors, difference in each direction, then sum."""
    def __init__(self):
        super().__init__()
        self.detect = nn.Linear(2, 2, bias=False)
        self.readout = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.detect.weight.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
            self.readout.weight.copy_(torch.tensor([[1., 1.]]))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.readout(torch.relu(self.detect(x))).squeeze(-1)

class GCDStep(nn.Module):
    """One transparent subtractive Euclidean update for nonnegative integer inputs."""
    def forward(self, xy):
        a, b = xy[..., 0], xy[..., 1]
        return torch.stack((torch.abs(a-b), torch.minimum(a, b)), dim=-1)

class GCDNet(nn.Module):
    """Repeated deterministic network block, no training and no Python gcd."""
    def __init__(self, iterations=1000):
        super().__init__()
        self.step = GCDStep()
        self.iterations = iterations

    def forward(self, xy):
        state = xy
        for _ in range(self.iterations):
            state = self.step(state)
        return state[..., 0]


def parity_data(n=128, seed=42):
    g=torch.Generator().manual_seed(seed)
    x=torch.randint(0,2,(n,2),generator=g).float()
    return x, (x[:,0] != x[:,1]).float()


def evaluate():
    x,y=parity_data(500)
    model=XorNet()
    with torch.no_grad():
        out=model(x)
    print(f'XOR exact match: {(out==y).float().mean().item():.3f} on {len(x)} synthetic samples')
    grid=torch.cartesian_prod(torch.arange(1,31),torch.arange(1,31)).float()
    import math
    expected=torch.tensor([math.gcd(int(a),int(b)) for a,b in grid.tolist()]).float()
    with torch.no_grad():
        prediction=GCDNet(iterations=100)(grid)
    print(f'GCD exact match: {(prediction==expected).float().mean().item():.3f} on {len(grid)} pairs (1..30)')
    print('GCD prediction example (84,36):',GCDNet(iterations=100)(torch.tensor([[84.,36.]])).item())

if __name__ == '__main__':
    evaluate()
