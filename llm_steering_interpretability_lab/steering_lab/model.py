"""Small controlled residual network, NOT a pretrained language model."""
import torch
from torch import nn
import torch.nn.functional as F

VOCAB = ["joyful", "gloomy", "tomato", "stone", "plain", "garden", "river", "city", "light", "quiet", "story", "day", "about", "the", "and", "is"]
TOKENS = {s: i for i, s in enumerate(VOCAB)}


class SwiGLU(nn.Module):
    def __init__(self, dim=48, inner=96):
        super().__init__()
        self.gate = nn.Linear(dim, inner)
        self.up = nn.Linear(dim, inner)
        self.tap = nn.Identity()
        self.down = nn.Linear(inner, dim)

    def forward(self, x):
        hidden = self.tap(F.silu(self.gate(x)) * self.up(x))
        return self.down(hidden)


class ResidualBlock(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, dim * 2)

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class ToyResidualPolicy(nn.Module):
    def __init__(self, dim=48, layers=4):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), dim)
        self.layers = nn.ModuleList([ResidualBlock(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.mood = nn.Linear(dim, 2)
        self.topic = nn.Linear(dim, 2)

    def forward(self, tokens):
        x = self.embed(tokens).mean(dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.mood(x), self.topic(x)


def make_batch(n=512, seed=0, split="train", neutral=False):
    """Balanced independent labels: 0=joyful, 1=gloomy; 0=stone, 1=tomato.

    The distributions in train/test use disjoint nuisance-token index ranges.
    This is an artificial concept experiment, not natural language comprehension.
    """
    g = torch.Generator().manual_seed(seed)
    mood = torch.randint(0, 2, (n,), generator=g)
    topic = torch.randint(0, 2, (n,), generator=g)
    tokens = torch.empty(n, 6, dtype=torch.long)
    tokens[:, 0] = 4 if neutral else torch.where(mood == 0, 0, 1)
    tokens[:, 1] = torch.where(topic == 0, 3, 2)
    if split == "train":
        tokens[:, 2:] = torch.randint(5, 11, (n, 4), generator=g)
    elif split == "test":
        tokens[:, 2:] = torch.randint(11, len(VOCAB), (n, 4), generator=g)
    else:
        raise ValueError(split)
    # shuffled positions prevent reliance on a fixed absolute position
    rand = torch.rand(n, 6, generator=g)
    tokens = tokens.gather(1, rand.argsort(dim=1))
    return tokens, mood, topic


def train_model(seed=0, steps=160, device="cpu"):
    torch.manual_seed(seed)
    model = ToyResidualPolicy().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.007)
    for step in range(steps):
        tokens, mood, topic = make_batch(256, seed=seed * 10_000 + step)
        tokens, mood, topic = tokens.to(device), mood.to(device), topic.to(device)
        opt.zero_grad(set_to_none=True)
        m, t = model(tokens)
        loss = F.cross_entropy(m, mood) + F.cross_entropy(t, topic)
        loss.backward()
        opt.step()
    model.eval()
    return model
