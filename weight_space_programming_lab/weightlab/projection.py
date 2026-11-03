"""Low-rank output-space projection for educational weight-editing experiments."""
from __future__ import annotations
import torch

@torch.no_grad()
def suppress_output_direction(weight: torch.Tensor, direction: torch.Tensor, *, columns=None, scale=1.0) -> torch.Tensor:
    """Edit W[d_out,d_in] along one output-space direction in selected input columns.

    scale=1 projects out the direction. scale>1 reverses its component.
    Returns a copy and never changes the supplied tensor.
    """
    if weight.ndim != 2 or direction.ndim != 1 or weight.shape[0] != direction.shape[0]:
        raise ValueError('direction must match matrix output dimensions')
    norm = direction.norm()
    if norm.item() < 1e-12: raise ValueError('direction must be nonzero')
    if scale < 0: raise ValueError('scale must be nonnegative')
    u=(direction/norm).to(device=weight.device,dtype=weight.dtype)
    result=weight.detach().clone()
    selected=list(range(weight.shape[1])) if columns is None else list(columns)
    if len(set(selected))!=len(selected) or any(not isinstance(c,int) or c<0 or c>=weight.shape[1] for c in selected):
        raise ValueError('columns must be unique valid indices')
    if not selected:return result
    block=result[:,selected].clone()
    result[:,selected] = block - scale*u.unsqueeze(1)*(u @ block).unsqueeze(0)
    return result
