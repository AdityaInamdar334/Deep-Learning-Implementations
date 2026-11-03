"""Train a tiny causal Decision Transformer on ONE fixed contextual policy.

This only verifies tokenization, masking, and supervised action prediction.
It is not an Atari or multi-task DT-Mem experiment.
"""
import argparse
import json
import torch
import torch.nn.functional as F
from .tiny_dt import TinyDecisionTransformer


def train(seed=13, steps=160, batch=48, horizon=6):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    teacher = torch.tensor([2, 0, 1, 3, 2, 1, 0, 3])
    model = TinyDecisionTransformer(n_states=8,n_actions=4,width=32,heads=4,layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    start_acc = None
    for step in range(steps):
        states = torch.randint(0, 8, (batch,horizon))
        actions = teacher[states]
        returns = torch.ones_like(states,dtype=torch.float)
        logits = model(returns,states,actions)
        if step == 0:
            start_acc = float((logits.argmax(-1)==actions).float().mean())
        loss = F.cross_entropy(logits.reshape(-1,4), actions.flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        states = torch.randint(0, 8, (256,horizon))
        actions=teacher[states]
        returns=torch.ones_like(states,dtype=torch.float)
        accuracy=float((model(returns,states,actions).argmax(-1)==actions).float().mean())
    return {'synthetic':True, 'single_fixed_mapping':True, 'initial_batch_accuracy':start_acc,
            'heldout_sequence_accuracy': accuracy, 'steps':steps,
            'params':sum(p.numel() for p in model.parameters())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=160);p.add_argument('--seed',type=int,default=13)
    a=p.parse_args(); print(json.dumps(train(seed=a.seed,steps=a.steps),indent=2))
