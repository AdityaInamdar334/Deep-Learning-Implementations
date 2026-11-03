"""A minimal, transparent content-addressed, soft write/read memory.

Prototypes are fixed by design, not learned. This is an explanatory memory
module inspired by the write/read concepts in Kang et al., ICML 2024.
"""
import torch
import torch.nn.functional as F


class SoftWorkingMemory:
    def __init__(self, num_contexts=8, num_actions=4, slots=8, temperature=30.0, device='cpu'):
        assert num_contexts > 0 and slots > 0 and num_actions > 1
        self.num_contexts = num_contexts
        self.num_actions = num_actions
        self.slots = slots
        self.device = device
        self.temperature = temperature
        assignment = torch.arange(num_contexts, device=device) % slots
        keys = F.one_hot(assignment, num_classes=slots).float().T
        # Each key is the centroid of its assigned one-hot context vectors.
        keys = keys / keys.sum(dim=1, keepdim=True).clamp_min(1)
        self.keys = keys
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.slots, self.num_actions, device=self.device)
        self.mass = torch.zeros(self.slots, device=self.device)

    def addressing(self, contexts):
        contexts = torch.as_tensor(contexts, device=self.device, dtype=torch.long).reshape(-1)
        assert bool(((0 <= contexts) & (contexts < self.num_contexts)).all())
        x = F.one_hot(contexts, num_classes=self.num_contexts).float()
        return torch.softmax(self.temperature * (x @ self.keys.T), dim=-1)

    def write(self, contexts, actions):
        contexts = torch.as_tensor(contexts, device=self.device, dtype=torch.long).reshape(-1)
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.long).reshape(-1)
        assert contexts.shape == actions.shape
        assert bool(((0 <= actions) & (actions < self.num_actions)).all())
        # Accumulate sequential observations in separate addressable slots.
        # Weighted add/read, rather than the paper's learned erase/write gate.
        for c, a in zip(contexts, actions):
            weight = self.addressing(c).squeeze(0)
            self.values += weight[:, None] * F.one_hot(a, self.num_actions).float()
            self.mass += weight

    def read(self, contexts):
        weights = self.addressing(contexts)
        slot_distribution = self.values / self.mass[:, None].clamp_min(1e-8)
        probabilities = weights @ slot_distribution
        coverage = weights @ self.mass.clamp(max=1.0)
        return probabilities, coverage

    def predict(self, contexts, fallback, min_coverage=0.5):
        probabilities, coverage = self.read(contexts)
        memory_preds = probabilities.argmax(dim=-1)
        fallback = torch.as_tensor(fallback, device=self.device, dtype=torch.long)
        return torch.where(coverage >= min_coverage, memory_preds, fallback)
