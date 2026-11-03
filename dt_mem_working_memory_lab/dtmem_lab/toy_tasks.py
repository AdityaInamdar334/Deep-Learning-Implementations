"""Task distributions with random task-specific mappings, for controlled tests."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_task(generator, contexts=8, actions=4):
    """Balanced labels with a new random mapping for every task."""
    assert contexts % actions == 0
    labels = torch.arange(actions).repeat(contexts // actions)
    return labels[torch.randperm(contexts, generator=generator)]


def draw_support(mapping, n, generator, corruption=0.0):
    contexts = torch.randint(0, len(mapping), (n,), generator=generator)
    labels = mapping[contexts].clone()
    if corruption > 0:
        changed = torch.rand(n, generator=generator) < corruption
        # Explicitly make corrupted labels wrong.
        labels[changed] = (labels[changed] + 1 + torch.randint(0, labels.max().item(), (int(changed.sum()),), generator=generator)) % (labels.max().item() + 1)
    return contexts, labels


def train_context_baseline(seed=0, contexts=8, actions=4, steps=180):
    """Train a context-only policy on many randomly remapped tasks.

    Since a held-out task mapping is independent, zero-shot predictions cannot
    discover its mapping without task-specific information.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1701)
    model = nn.Sequential(nn.Linear(contexts, 32), nn.ReLU(), nn.Linear(32, actions))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    model.train()
    for step in range(steps):
        task = sample_task(generator, contexts, actions)
        sample_ctx = torch.randint(0, contexts, (64,), generator=generator)
        logits = model(F.one_hot(sample_ctx, num_classes=contexts).float())
        loss = F.cross_entropy(logits, task[sample_ctx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def baseline_predict(model, contexts, n_contexts=8):
    return model(F.one_hot(contexts.long(), num_classes=n_contexts).float()).argmax(-1)
