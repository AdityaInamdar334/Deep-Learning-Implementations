"""SwiGLU from scratch: geometry, matched-parameter FFNs, and toy regression.
Only requires torch and matplotlib (matplotlib for --plots).
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

class PlainFFN(nn.Module):
    def __init__(self, d, hidden, activation='relu'):
        super().__init__()
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.activation = activation
    def forward(self, x):
        h = self.up(x)
        h = F.relu(h) if self.activation == 'relu' else F.silu(h)
        return self.down(h)

class SwiGLUFFN(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class ToyReLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,width), nn.ReLU(), nn.Linear(width,1))
    def forward(self, x): return self.net(x).squeeze(-1)

class ToySwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(2,1,bias=False)
        self.value = nn.Linear(2,1,bias=False)
    def forward(self, x):
        return (F.silu(self.gate(x))*self.value(x)).squeeze(-1)

def target(x):
    # Interpretable synthetic task: x1 controls gate; x2 is signed value.
    return F.silu(2.5*x[:,0])*x[:,1]

def train_toy(model, x_train, y_train, x_test, y_test, steps=700, lr=0.025):
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(x_train),y_train)
        loss.backward()
        opt.step()
    with torch.no_grad(): return F.mse_loss(model(x_test),y_test).item()

def run(seed=11, steps=700, plots=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    x_train = (torch.rand(1500,2)*4-2)
    x_test = (torch.rand(500,2)*4-2)
    y_train, y_test = target(x_train), target(x_test)
    models = {'ReLU width=1':ToyReLU(1), 'ReLU width=4':ToyReLU(4), 'ReLU width=16':ToyReLU(16), 'SwiGLU width=1':ToySwiGLU()}
    # A deliberately favorable, transparent initialization: highlights representation,
    # not an optimizer win. Random-start SwiGLU can fail on this nonconvex toy.
    with torch.no_grad():
        models['SwiGLU width=1'].gate.weight.copy_(torch.tensor([[1.0,0.0]]))
        models['SwiGLU width=1'].value.weight.copy_(torch.tensor([[0.0,0.5]]))
    print('Target function: SiLU(2.5*x1) * x2 (synthetic, constructed for gating)')
    for name, model in models.items():
        error = train_toy(model,x_train,y_train,x_test,y_test,steps=steps)
        n = sum(p.numel() for p in model.parameters())
        print(f'{name:17s} parameters={n:3d}  heldout_MSE={error:.6f}')
    d=96
    plain=PlainFFN(d,4*d)
    gated=SwiGLUFFN(d,8*d//3)
    assert sum(p.numel() for p in plain.parameters()) == sum(p.numel() for p in gated.parameters())
    print(f'Equal-budget FFNs d={d}: {sum(p.numel() for p in plain.parameters()):,} parameters each')
    if plots:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        out = Path(__file__).resolve().parent/'figures'; out.mkdir(exist_ok=True)
        v=torch.linspace(-5,5,501)
        plt.figure(figsize=(7,4))
        plt.plot(v.numpy(),F.relu(v).numpy(),label='ReLU')
        plt.plot(v.numpy(),F.silu(v).numpy(),label='SiLU / Swish')
        plt.xlabel('Preactivation'); plt.ylabel('Output'); plt.legend(); plt.tight_layout();plt.savefig(out/'relu_vs_silu.png',dpi=160);plt.close()
        z=torch.linspace(-2,2,90); xx,yy=torch.meshgrid(z,z,indexing='ij'); grid=torch.stack((xx.flatten(),yy.flatten()),dim=-1)
        actual=target(grid).view(xx.shape)
        fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d');ax.plot_surface(xx.numpy(),yy.numpy(),actual.numpy(),cmap='viridis',alpha=.9,linewidth=0);ax.set(xlabel='x1: gate feature',ylabel='x2: value feature',zlabel='target',title='A conditional feature interaction');plt.tight_layout();plt.savefig(out/'conditional_surface.png',dpi=160);plt.close()
        fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d');ax.plot_surface(xx.numpy(),yy.numpy(),models['SwiGLU width=1'](grid).detach().view(xx.shape).numpy(),cmap='viridis',alpha=.9,linewidth=0);ax.set(xlabel='x1',ylabel='x2',zlabel='prediction',title='One learned SwiGLU unit');plt.tight_layout();plt.savefig(out/'learned_swiglu_surface.png',dpi=160);plt.close()
        print(f'Plots written to {out}')

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=700);p.add_argument('--plots',action='store_true');args=p.parse_args();run(steps=args.steps,plots=args.plots)
