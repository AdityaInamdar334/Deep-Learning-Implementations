"""Small paired supervision demo; NOT an experimental reproduction of the paper."""
import copy
import hashlib
import random
import numpy as np

LABELS = ('correct', 'fixed_random', 'changing_random')

def seed_all(seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def gaussian_data(seed=7, n_train=360, n_test=180):
    """Three synthetic 2D classes. No Fashion-MNIST downloads required."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0,-0.7], [2.0,-0.7], [0.0,2.3]])
    def sample(n):
        y = np.tile(np.arange(3), (n + 2)//3)[:n]
        rng.shuffle(y)
        x = centers[y] + rng.normal(0, 0.78, size=(n,2))
        return x.astype('float32'), y.astype('int64')
    return *sample(n_train), *sample(n_test)

def permuted_labels(labels, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(labels)[rng.permutation(len(labels))]

def label_for_epoch(original, condition, base_seed, epoch):
    if condition == 'correct':
        return np.asarray(original).copy()
    if condition == 'fixed_random':
        return permuted_labels(original, base_seed + 1000)
    if condition == 'changing_random':
        return permuted_labels(original, base_seed + 1000 + epoch)
    raise ValueError(f'Unknown condition: {condition}')

def label_digest(labels):
    return hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()

def build_mlp(input_dim=2, width=12, classes=3):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(input_dim,width), nn.ReLU(),
                         nn.Linear(width,width), nn.ReLU(),
                         nn.Linear(width,classes))

def training_run(seed=42, epochs=12, output_dir=None):
    """Train paired cases on a tiny toy dataset; saves models + true-label metrics.

    Explicitly not the paper's 784->... Fashion-MNIST experiment.
    """
    import torch
    from pathlib import Path
    from .topology import weight_cloud, rips_h0_h1, summarize
    x, y, tx, ty = gaussian_data(seed)
    xx = torch.tensor(x)
    yy = torch.tensor(y)
    testx = torch.tensor(tx)
    testy = torch.tensor(ty)
    seed_all(seed)
    initial = build_mlp()
    starting_state = copy.deepcopy(initial.state_dict())
    results = {}
    for condition in LABELS:
        model = build_mlp()
        model.load_state_dict(starting_state)
        opt = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.001)
        checkpoints = []
        for epoch in range(epochs + 1):
            if epoch > 0:
                chosen = torch.tensor(label_for_epoch(y, condition, seed, epoch - 1))
                perm = np.random.default_rng(seed * 1000 + epoch).permutation(len(y))
                model.train()
                for start in range(0, len(y), 72):
                    ix = torch.as_tensor(perm[start:start+72])
                    opt.zero_grad()
                    loss = torch.nn.functional.cross_entropy(model(xx[ix]), chosen[ix])
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                train_acc = float((model(xx).argmax(dim=1) == yy).float().mean())
                test_acc = float((model(testx).argmax(dim=1) == testy).float().mean())
                w = model[0].weight.detach().numpy().copy()
            cloud = weight_cloud(w, count=12, normalized=True)
            diagrams = rips_h0_h1(cloud)
            checkpoints.append({'epoch': epoch, 'train_accuracy_on_true_labels': train_acc,
                 'test_accuracy_on_true_labels': test_acc,
                 'h0_total': summarize(diagrams[0])['total'],
                 'h1_total': summarize(diagrams[1])['total'],
                 'h1_count': summarize(diagrams[1])['count']})
        results[condition] = checkpoints
    return results
