"""Synthetic recurrent token corpora with separate RNG streams for tests."""
import torch


def repeat_batch(batch: int, length: int, period: int=4, vocab: int=32,
                 corruption: float=0.07, seed: int=0, device='cpu') -> torch.Tensor:
    """Return (batch, length+1) discrete sequences.

    Next-token task is to predict seq[t+1] from seq[:t+1]. Most tokens copy
    period positions back, some are replaced. No train/val/test overlap when
    seeds differ. This is NOT a natural language benchmark.
    """
    if period < 1 or period > length: raise ValueError('invalid period')
    g = torch.Generator(device='cpu').manual_seed(seed)
    init = torch.randint(0,vocab,(batch,period),generator=g)
    seq = [init[:,j] for j in range(period)]
    for t in range(period,length+1):
        prev = seq[t-period]
        repl = torch.randint(0,vocab,(batch,),generator=g)
        choose = torch.rand(batch,generator=g)<corruption
        seq.append(torch.where(choose,repl,prev))
    return torch.stack(seq,dim=1).to(device)
