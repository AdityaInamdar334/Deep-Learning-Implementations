"""Tiny autoregressive Hourglass and vanilla baselines (batch, tokens, channels).

Crucial invariant: a pooled representation broadcast to output position t must
only depend on original input positions <= t. Pooling itself is not causal.
The pre-pooling shift by k-1 is what makes the broadcast safe.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def causal_shift(x: torch.Tensor, amount: int) -> torch.Tensor:
    """Shift token sequence to the right; fill with zeros, never wrap."""
    if amount < 0: raise ValueError('amount must be nonnegative')
    if amount == 0: return x
    zeros = x.new_zeros((*x.shape[:-2], amount, x.shape[-1]))
    return torch.cat((zeros,x[..., :-amount, :]), dim=-2) if amount < x.shape[-2] else x.new_zeros(x.shape)


def upsample_repeat(short: torch.Tensor, factor: int, target_length: int) -> torch.Tensor:
    if factor < 1: raise ValueError('factor must be >=1')
    return torch.repeat_interleave(short, factor, dim=-2)[..., :target_length, :]


class AverageShortener(nn.Module):
    def forward(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        if factor < 1: raise ValueError('factor must be >=1')
        # avg_pool1d with ceil_mode preserves trailing partial group.
        return F.avg_pool1d(x.transpose(1,2), kernel_size=factor,
                            stride=factor, ceil_mode=True).transpose(1,2)


class LinearShortener(nn.Module):
    def __init__(self, width: int, factor: int):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(width * factor, width)
    def forward(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        if factor != self.factor: raise ValueError('linear shortener has fixed factor')
        batch,length,width = x.shape
        missing = (-length) % factor
        if missing:
            x = F.pad(x, (0,0,0,missing))
        return self.proj(x.reshape(batch,(length+missing)//factor,width*factor))


class LinearUpsampler(nn.Module):
    def __init__(self, width: int, factor: int):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(width, width * factor)
        # Small initial contribution leaves the full-resolution path prominent.
        nn.init.normal_(self.proj.weight, mean=0, std=0.01)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x: torch.Tensor, target_length: int, factor: int) -> torch.Tensor:
        if factor != self.factor: raise ValueError('linear upsampler has fixed factor')
        b,n,d = x.shape
        return self.proj(x).reshape(b,n*factor,d)[:,:target_length,:]


class TinyCausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, expansion: int=2):
        super().__init__()
        if width % heads != 0: raise ValueError('width must be divisible by heads')
        self.heads = heads
        self.dk = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.o = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width,width*expansion),nn.GELU(),nn.Linear(width*expansion,width))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b,n,d = x.shape
        q,k,v = self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,self.dk).unbind(dim=2)
        q,k,v = [part.transpose(1,2) for part in (q,k,v)]
        a = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x = x + self.o(a.transpose(1,2).contiguous().reshape(b,n,d))
        return x + self.ff(self.norm2(x))


class HourglassCore(nn.Module):
    """Recursive 1-pre / middle / 1-post hourglass; optional learned pooling."""
    def __init__(self, width: int, heads: int, factors: tuple[int,...], middle_layers: int=2,
                 pooling: str='average', do_shift: bool=True, skip: bool=True, pre: bool=True, post: bool=True):
        super().__init__()
        if not factors or any(f < 2 for f in factors): raise ValueError('shortening factors must be >=2')
        self.factor = factors[0]
        self.pooling = pooling
        self.do_shift = do_shift
        self.skip = skip
        self.pre = TinyCausalBlock(width,heads) if pre else nn.Identity()
        self.post = TinyCausalBlock(width,heads) if post else nn.Identity()
        if pooling == 'average':
            self.down = AverageShortener()
            self.up = None
        elif pooling == 'linear':
            self.down = LinearShortener(width, self.factor)
            self.up = LinearUpsampler(width, self.factor)
        else: raise ValueError('pooling must be average or linear')
        self.inner = (HourglassCore(width,heads,factors[1:],middle_layers,pooling,do_shift,skip,pre,post)
                      if len(factors)>1 else nn.Sequential(*(TinyCausalBlock(width,heads) for _ in range(middle_layers))))
    def forward(self, x: torch.Tensor, override_factor: int | None=None) -> torch.Tensor:
        k = override_factor or self.factor
        if self.pooling=='linear' and k!=self.factor: raise ValueError('fixed-factor learned resampling')
        dense = self.pre(x)
        short_in = causal_shift(dense,k-1) if self.do_shift else dense
        short = self.down(short_in,k)
        # Dynamic factor applies to first shortening only. Recursive lower levels remain fixed.
        short = self.inner(short)
        up = (self.up(short,dense.shape[1],k) if self.up is not None
              else upsample_repeat(short,k,dense.shape[1]))
        out = dense + up if self.skip else up
        return self.post(out)


def sinusoidal_positions(length: int, width: int, device, dtype):
    pos = torch.arange(length,device=device,dtype=torch.float32).unsqueeze(1)
    freqs = torch.exp(torch.arange(0,width,2,device=device,dtype=torch.float32)*(-math.log(10000.0)/width))
    pe = torch.zeros(length,width,device=device,dtype=torch.float32)
    pe[:,0::2] = torch.sin(pos*freqs)
    pe[:,1::2] = torch.cos(pos*freqs[:pe[:,1::2].shape[1]])
    return pe.to(dtype)

class TokenLM(nn.Module):
    def __init__(self, vocab: int, width: int):
        super().__init__()
        self.emb = nn.Embedding(vocab,width)
        self.out_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width,vocab)
        self.width = width
    def encode(self, tokens):
        return self.emb(tokens)* (self.width**0.5) + sinusoidal_positions(tokens.size(1),self.width,tokens.device,self.emb.weight.dtype)[None]
    def decode(self, h):
        return self.head(self.out_norm(h))

class VanillaLM(TokenLM):
    def __init__(self,vocab=32,width=32,heads=4,layers=4):
        super().__init__(vocab,width)
        self.blocks = nn.Sequential(*(TinyCausalBlock(width,heads) for _ in range(layers)))
    def forward(self,tokens):
        return self.decode(self.blocks(self.encode(tokens)))

class HourglassLM(TokenLM):
    def __init__(self,vocab=32,width=32,heads=4,factors=(3,),middle_layers=2,
                 pooling='average',do_shift=True,skip=True,pre=True,post=True):
        super().__init__(vocab,width)
        self.core = HourglassCore(width,heads,tuple(factors),middle_layers,pooling,do_shift,skip,pre,post)
    def forward(self,tokens,override_factor=None):
        return self.decode(self.core(self.encode(tokens), override_factor=override_factor))
