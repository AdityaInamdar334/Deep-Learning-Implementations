"""A tiny return-conditioned causal DT for architecture teaching, NOT DT-Mem.

This class is independent of the benchmark's fixed-key memory ablations.
"""
import torch
from torch import nn


class TinyDecisionTransformer(nn.Module):
    def __init__(self, n_states=8, n_actions=4, width=32, heads=4, layers=1):
        super().__init__()
        self.n_actions = n_actions
        self.state = nn.Embedding(n_states, width)
        self.action = nn.Embedding(n_actions + 1, width)
        self.ret = nn.Linear(1, width)
        self.pos = nn.Embedding(512, width)
        block = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width,
                                           dropout=0.0, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, n_actions)

    def forward(self, returns, states, actions):
        """All arguments [batch,time]. Action at t must NOT leak to prediction t.

        A target action token occurs AFTER the corresponding state token and
        the causal mask prevents the state token attending to that action.
        """
        if not (returns.shape == states.shape == actions.shape):
            raise ValueError('returns, states, actions need identical [batch,time] shapes')
        b, t = states.shape
        if 3*t > 512:
            raise ValueError('sequence too long for position embeddings')
        r = self.ret(returns.float().unsqueeze(-1))
        s = self.state(states.long())
        a = self.action(actions.long())
        x = torch.stack((r, s, a), dim=2).reshape(b, 3*t, -1)
        x = x + self.pos(torch.arange(3*t, device=x.device))[None, :, :]
        # True means forbidden for subsequent positions.
        forbidden = torch.triu(torch.ones(3*t, 3*t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.transformer(x, mask=forbidden)
        return self.head(h[:, 1::3, :])
