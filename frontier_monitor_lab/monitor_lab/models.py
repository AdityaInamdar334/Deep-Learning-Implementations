"""PyTorch proxy activation probe and lightweight bag-of-tokens reviewer."""
from collections import Counter
import re
import torch
from torch import nn

torch.set_num_threads(1)


def tokens(text):
    return re.findall(r'[a-z0-9_]+', text.lower())


def build_vocab(cases):
    counter = Counter(w for case in cases for w in tokens(review_text(case)))
    return {'<unk>': 0, **{word: i+1 for i, (word, _) in enumerate(sorted(counter.items()))}}


def review_text(case, text_only=False):
    if text_only:
        return case.narrative
    return f'{case.narrative} action_{case.tool_action} resource_{case.resource} task_{case.allowed_action}'


def features_text(cases, vocab, text_only=False):
    matrix = torch.zeros((len(cases), len(vocab)), dtype=torch.float32)
    for row, case in enumerate(cases):
        for token in tokens(review_text(case, text_only=text_only)):
            matrix[row, vocab.get(token, 0)] += 1
    # Binary term presence, not token count or variable-length sentence length.
    return (matrix > 0).float()


def features_probe(cases):
    return torch.tensor([list(c.proxy_activation) for c in cases], dtype=torch.float32)


class LinearProbe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).view(-1)


class TextReviewer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 48), nn.ReLU(), nn.Linear(48, 1))

    def forward(self, x):
        return self.net(x).view(-1)


def train_model(model, x, labels, seed, epochs=75, lr=0.035):
    torch.manual_seed(seed)
    y = torch.tensor(labels, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def probabilities(model, x):
    return torch.sigmoid(model(x)).tolist()


def recall_threshold(scores, labels, target_recall=0.90):
    """Choose highest threshold meeting specified TPR on held-out validation positives."""
    positives = sorted((score for score, label in zip(scores, labels) if label), reverse=True)
    if not positives:
        raise ValueError('No positives in validation set')
    import math
    k = math.ceil(target_recall * len(positives))
    return max(0., positives[k - 1] - 1e-8)
