"""Constrained no-gradient edits, simple controls, and exact model-diff checks."""
from __future__ import annotations
import copy
import torch
from .core import TinyPolicy


def propose_edit(model: TinyPolicy, method: str, *, seed: int=0) -> TinyPolicy:
    """No gradients, training data or labels. This is a programmed policy, NOT an LLM."""
    m = copy.deepcopy(model)
    w = m.fc2.weight
    with torch.no_grad():
        if method == 'original':
            return m
        if method == 'semantic':
            # External task semantics say that action index 2 is a stall.
            # Remove only the two nuisance-induced output links.
            w[2, 2] = 0
            w[2, 3] = 0
        elif method == 'half_semantic':
            w[2,2] = 0
        elif method == 'top_magnitude':
            # A simple competing baseline. In THIS injected-outlier case it succeeds.
            flat = w.view(-1)
            ix = torch.topk(flat.abs(), k=2).indices
            flat[ix] = 0
        elif method == 'random_two':
            g = torch.Generator().manual_seed(seed)
            flat = w.view(-1)
            nz = torch.nonzero(flat != 0).flatten()
            ix = nz[torch.randperm(nz.numel(), generator=g)[:2]]
            flat[ix] = 0
        elif method == 'global_shrink':
            w.mul_(0.5)
        elif method == 'wrong_path':
            w[0,0] = 0
            w[1,1] = 0
        else:
            raise ValueError(f"Unknown edit method: {method}")
    return m


def diff_audit(before: TinyPolicy, after: TinyPolicy, *, tol=0) -> dict:
    b = before.state_dict(); a = after.state_dict()
    if set(a) != set(b): raise ValueError("Model structures do not match")
    changed = total = 0; names={}; max_abs=0.
    for name in sorted(a):
        if a[name].shape != b[name].shape:
            raise ValueError("Model shapes do not match")
        diff=(a[name]-b[name]).abs()
        count=int((diff > tol).sum())
        names[name]=count; changed += count; total += diff.numel()
        max_abs=max(max_abs,float(diff.max().item()))
    return {'changed':changed,'total':total,'fraction':changed/total,'by_tensor':names,'max_absolute_delta':max_abs}


def hidden_circuit(model: TinyPolicy) -> dict:
    """Return the first-layer source channel of each nonzero hidden unit."""
    w=model.fc1.weight.detach()
    return {int(i): {'source':int(w[i].abs().argmax()),'sign':float(torch.sign(w[i,w[i].abs().argmax()])), 'strength':float(w[i].abs().max())}
            for i in range(w.shape[0]) if w[i].abs().max()>0}
