"""Pure-NumPy toy top-k MoE. It demonstrates routing and cost accounting.

This is a single MoE feed-forward block, not a transformer. It intentionally
uses small matrices and unoptimized per-token dispatch for transparency.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Cost:
    total_parameters: int
    active_parameters_per_token: int
    estimated_macs_per_token: int
    weight_bytes_float32: int


class ToyMoE:
    def __init__(self, n_experts=8, hidden=64, intermediate=128, seed=7):
        if min(n_experts, hidden, intermediate) < 1:
            raise ValueError('Every dimension must be positive')
        self.n_experts = n_experts
        self.hidden = hidden
        self.intermediate = intermediate
        rng = np.random.default_rng(seed)
        self.router = rng.normal(0, 1 / np.sqrt(hidden), (hidden, n_experts)).astype(np.float32)
        self.w_up = rng.normal(0, 1 / np.sqrt(hidden), (n_experts, hidden, intermediate)).astype(np.float32)
        self.w_down = rng.normal(0, 1 / np.sqrt(intermediate), (n_experts, intermediate, hidden)).astype(np.float32)

    def cost(self, k):
        if not (1 <= k <= self.n_experts):
            raise ValueError('top_k must lie between 1 and n_experts')
        router_p = self.hidden * self.n_experts
        expert_p = 2 * self.hidden * self.intermediate
        return Cost(
            total_parameters=router_p + self.n_experts * expert_p,
            active_parameters_per_token=router_p + k * expert_p,
            estimated_macs_per_token=router_p + k * expert_p,
            weight_bytes_float32=self.router.nbytes + self.w_up.nbytes + self.w_down.nbytes,
        )

    def routes(self, x, top_k=2):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.hidden:
            raise ValueError(f'Expected x with shape (tokens, {self.hidden})')
        if not (1 <= top_k <= self.n_experts):
            raise ValueError('invalid top_k')
        logits = x @ self.router
        # Stable, deterministic selection for fixed inputs; ties are rare here.
        ids = np.argsort(-logits, axis=-1, kind='stable')[:, :top_k]
        chosen = np.take_along_axis(logits, ids, axis=-1)
        exps = np.exp(chosen - np.max(chosen, axis=-1, keepdims=True))
        weights = exps / exps.sum(axis=-1, keepdims=True)
        return ids, weights

    def expert(self, token, expert_id):
        return np.maximum(token @ self.w_up[expert_id], 0) @ self.w_down[expert_id]

    def forward(self, x, top_k=2):
        x = np.asarray(x, dtype=np.float32)
        ids, weights = self.routes(x, top_k)
        out = np.zeros_like(x)
        # This intentionally explicit dispatch has Python overhead.
        for row in range(len(x)):
            for col, expert_id in enumerate(ids[row]):
                out[row] += weights[row, col] * self.expert(x[row], int(expert_id))
        return out

    def dense_reference(self, x):
        """All experts, softmax over all; equal to forward(x, top_k=n_experts)."""
        x = np.asarray(x, dtype=np.float32)
        logits = x @ self.router
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        out = np.zeros_like(x)
        for row in range(len(x)):
            for expert_id in range(self.n_experts):
                out[row] += p[row, expert_id] * self.expert(x[row], expert_id)
        return out
