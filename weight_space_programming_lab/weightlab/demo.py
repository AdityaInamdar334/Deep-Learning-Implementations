from __future__ import annotations
import torch
from .core import make_policy, compile_xor, permute_hidden, inspect_axis_naive, inspect_axis_invariant
from .editing import propose_edit,diff_audit,hidden_circuit
from .evaluate import classification_accuracy,navigation_success

def main():
    x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    print('XOR from manually compiled weights:')
    print(list(zip(x.int().tolist(), compile_xor()(x).detach().flatten().int().tolist())))
    m=make_policy(axis=2,nuisance=3)
    g=torch.Generator().manual_seed(42)
    pm=permute_hidden(m,torch.randperm(16,generator=g))
    test=torch.randn(100,4,generator=g)
    print('Same function after permutation:',torch.allclose(m(test),pm(test)))
    print('Naive vs invariant weight-reader:',inspect_axis_naive(pm),inspect_axis_invariant(pm))
    flawed=make_policy(broken=True)
    print('A small deterministic network is corrupted at two output connections')
    print('Visible weight path:',hidden_circuit(flawed))
    for method in ['original','semantic','half_semantic','top_magnitude','random_two','wrong_path']:
        edited=propose_edit(flawed,method,seed=21)
        print(method,'audit',diff_audit(flawed,edited),'holdout',classification_accuracy(edited,seed=76),'rollout',navigation_success(edited,seed=77))
if __name__ == '__main__': main()
