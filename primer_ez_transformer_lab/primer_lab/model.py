"""Independent educational implementation of Primer-EZ's two main primitives.

Not Google's official implementation. All tensors are batch-first.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


class SquaredReLU(nn.Module):
    """Elementwise max(x, 0)**2. The function has no learnable parameters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(x)
        return y * y


class CausalDepthwiseConv1d(nn.Module):
    """Apply kernel-size k convolution to each projected head channel, without future leakage.

    x: [batch, heads, sequence, head_dim].
    shared_across_heads=True matches the kernel-sharing layout of labml.ai's
    reference implementation; False learns distinct kernels for every head.
    """

    def __init__(self, heads: int, head_dim: int, kernel_size: int = 3,
                 shared_across_heads: bool = True):
        super().__init__()
        if heads <= 0 or head_dim <= 0 or kernel_size < 1:
            raise ValueError('positive heads/head_dim and kernel_size required')
        self.heads = heads
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.shared_across_heads = shared_across_heads
        channels = head_dim if shared_across_heads else heads * head_dim
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              groups=channels, bias=True, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.heads or x.shape[-1] != self.head_dim:
            raise ValueError('expected [batch, heads, seq, head_dim]')
        b, h, t, d = x.shape
        if self.shared_across_heads:
            packed = x.permute(0, 1, 3, 2).reshape(b * h, d, t)
        else:
            packed = x.permute(0, 1, 3, 2).reshape(b, h * d, t)
        padded = F.pad(packed, (self.kernel_size - 1, 0))
        mixed = self.conv(padded)
        if self.shared_across_heads:
            return mixed.reshape(b, h, d, t).permute(0, 1, 3, 2).contiguous()
        return mixed.reshape(b, h, d, t).permute(0, 1, 3, 2).contiguous()


class CausalSelfAttention(nn.Module):
    """Explicit, auditable attention. Uses plain matmul, not fused SDPA."""

    def __init__(self, d_model: int, heads: int, use_dconv: bool = False,
                 shared_across_heads: bool = True, kernel_size: int = 3,
                 conv_targets: str = 'qkv'):
        super().__init__()
        if d_model % heads:
            raise ValueError('d_model must be divisible by heads')
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.use_dconv = use_dconv
        if not set(conv_targets) <= set('qkv') or not conv_targets:
            raise ValueError('conv_targets must be a nonempty subset of qkv')
        self.conv_targets = conv_targets
        if use_dconv:
            for target in conv_targets:
                setattr(self, target + '_conv', CausalDepthwiseConv1d(
                    heads, self.head_dim, kernel_size, shared_across_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = [u.permute(0, 2, 1, 3) for u in (q, k, v)]
        if self.use_dconv:
            if 'q' in self.conv_targets:
                q = self.q_conv(q)
            if 'k' in self.conv_targets:
                k = self.k_conv(k)
            if 'v' in self.conv_targets:
                v = self.v_conv(v)
        score = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        future = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        score = score.masked_fill(future, torch.finfo(score.dtype).min)
        att = score.softmax(dim=-1)
        value = (att @ v).transpose(1, 2).reshape(b, t, c)
        return self.out(value)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int,
                 activation: str = 'relu', use_dconv: bool = False,
                 shared_across_heads: bool = True, conv_targets: str = 'qkv'):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, heads, use_dconv, shared_across_heads,
                                        conv_targets=conv_targets)
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'squared_relu':
            act = SquaredReLU()
        elif activation == 'gelu':
            act = nn.GELU()
        else:
            raise ValueError(f'unsupported activation: {activation}')
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), act, nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16
    context: int = 24
    d_model: int = 48
    heads: int = 4
    layers: int = 2
    d_ff: int = 144
    activation: str = 'relu'
    use_dconv: bool = False
    shared_across_heads: bool = True
    conv_targets: str = 'qkv'


class TinyLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.positions = nn.Embedding(config.context, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.heads, config.d_ff,
                             config.activation, config.use_dconv, config.shared_across_heads,
                             config.conv_targets)
            for _ in range(config.layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError('tokens must be [batch, time]')
        b, t = tokens.shape
        if t > self.config.context:
            raise ValueError('sequence exceeds trained context')
        pos = torch.arange(t, device=tokens.device)
        x = self.embed(tokens) * (self.config.d_model ** 0.5) + self.positions(pos)
        # Keep embedding variance stable at tiny scale; gradient and comparison are shared.
        x = x / (self.config.d_model ** 0.5)
        for layer in self.blocks:
            x = layer(x)
        return self.output(self.norm(x))


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


ABLATIONS = {
    'baseline_relu': dict(activation='relu', use_dconv=False),
    'squared_relu_only': dict(activation='squared_relu', use_dconv=False),
    'dconv_only': dict(activation='relu', use_dconv=True),
    'primer_ez': dict(activation='squared_relu', use_dconv=True),
    'gelu_baseline': dict(activation='gelu', use_dconv=False),
    'primer_per_head': dict(activation='squared_relu', use_dconv=True, shared_across_heads=False),
    'q_only': dict(activation='relu', use_dconv=True, conv_targets='q'),
    'k_only': dict(activation='relu', use_dconv=True, conv_targets='k'),
    'v_only': dict(activation='relu', use_dconv=True, conv_targets='v'),
}


def make_model(name: str, **kwargs) -> TinyLanguageModel:
    if name not in ABLATIONS:
        raise ValueError(f'unknown configuration: {name}')
    return TinyLanguageModel(ModelConfig(**{**kwargs, **ABLATIONS[name]}))
