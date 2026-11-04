"""NeoX-style decoder with MHA, GQA, and MQA; educational implementation."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab_size: int = 256
    hidden_size: int = 128
    num_heads: int = 8
    num_kv_heads: int = 2
    backend: str = "repeat"
    num_layers: int = 4
    intermediate_size: int = 512
    rotary_pct: float = 0.25
    rope_base: float = 10000.0
    max_seq_len: int = 256
    layer_norm_eps: float = 1e-5

    def __post_init__(self):
        if min(self.vocab_size, self.hidden_size, self.num_heads,
               self.num_layers, self.intermediate_size, self.max_seq_len) < 1:
            raise ValueError('Dimensions must be positive')
        if self.num_kv_heads < 1 or self.num_heads % self.num_kv_heads:
            raise ValueError('num_heads must be divisible by positive num_kv_heads')
        if self.backend not in ('repeat', 'grouped', 'native'):
            raise ValueError('Unknown attention backend')
        if self.hidden_size % self.num_heads:
            raise ValueError('hidden_size must be divisible by num_heads')
        if not 0 < self.rotary_pct <= 1 or self.rope_base <= 0:
            raise ValueError('Invalid RoPE settings')
        rotary_dim = int(self.hidden_size // self.num_heads * self.rotary_pct)
        if rotary_dim < 2 or rotary_dim % 2:
            raise ValueError('Rotary dimension must be positive and even')

class Rotary(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.dim, self.base = dim, base

    def forward(self, x, offset):
        # x: batch, heads, new_tokens, head_dim. Angles are computed in FP32.
        positions = torch.arange(offset, offset + x.shape[-2],
                                 device=x.device, dtype=torch.float32)
        inv_freq = self.base ** (-torch.arange(0, self.dim, 2,
                        device=x.device, dtype=torch.float32) / self.dim)
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        rotated, rest = x[..., :self.dim], x[..., self.dim:]
        first, second = rotated.chunk(2, dim=-1)
        half_turn = torch.cat((-second, first), dim=-1)
        return torch.cat((rotated * cos + half_turn * sin, rest), dim=-1)

def grouped_attention(q, k, v, allowed):
    """Readable reference: no explicit K/V repetition; materializes scores."""
    batch, hq, length, dim = q.shape
    hkv, source = k.shape[1], k.shape[-2]
    queries = q.reshape(batch, hkv, hq // hkv, length, dim)
    scores = torch.einsum('bgrtd,bgsd->bgrts', queries, k) * (dim ** -0.5)
    scores = scores.masked_fill(~allowed, float('-inf'))
    weights = scores.float().softmax(-1).to(q.dtype)
    output = torch.einsum('bgrts,bgsd->bgrtd', weights, v)
    return output.reshape(batch, hq, length, dim)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.kv_heads = c.num_heads, c.num_kv_heads
        self.head_dim = c.hidden_size // c.num_heads
        self.backend = c.backend
        self.q_proj = nn.Linear(c.hidden_size, self.heads * self.head_dim)
        self.k_proj = nn.Linear(c.hidden_size, self.kv_heads * self.head_dim)
        self.v_proj = nn.Linear(c.hidden_size, self.kv_heads * self.head_dim)
        self.out = nn.Linear(c.hidden_size, c.hidden_size)
        self.rope = Rotary(int(self.head_dim * c.rotary_pct), c.rope_base)

    def forward(self, x, past=None, use_cache=False):
        batch, length, width = x.shape
        def split(projection, heads):
            return projection(x).view(batch, length, heads, self.head_dim).transpose(1, 2)
        q = split(self.q_proj, self.heads)
        k, v = split(self.k_proj, self.kv_heads), split(self.v_proj, self.kv_heads)
        offset = 0 if past is None else past[0].shape[-2]
        q, k = self.rope(q, offset), self.rope(k, offset)
        if past is not None:
            expected = (batch, self.kv_heads, offset, self.head_dim)
            if tuple(past[0].shape) != expected or tuple(past[1].shape) != expected:
                raise ValueError('Cache has wrong batch/head/sequence/dimension shape')
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        # Save compact K/V BEFORE any temporary repetition for the kernel.
        present = (k, v) if use_cache else None
        query_positions = offset + torch.arange(length, device=x.device)
        key_positions = torch.arange(k.shape[-2], device=x.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        if self.backend == 'grouped':
            y = grouped_attention(q, k, v, allowed)
        elif self.backend == 'native':
            # No hidden fallback: unsupported kernels should fail visibly.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                    dropout_p=0.0, is_causal=False, enable_gqa=True)
        else:
            repeats = self.heads // self.kv_heads
            expanded_k = k.repeat_interleave(repeats, dim=1)
            expanded_v = v.repeat_interleave(repeats, dim=1)
            y = F.scaled_dot_product_attention(q, expanded_k, expanded_v,
                    attn_mask=allowed, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.out(y), present

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.mlp_norm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c.hidden_size, c.intermediate_size),
                                 nn.GELU(),
                                 nn.Linear(c.intermediate_size, c.hidden_size))

    def forward(self, x, past=None, use_cache=False):
        attention, present = self.attn(self.attn_norm(x), past, use_cache)
        return x + attention + self.mlp(self.mlp_norm(x)), present

class TinyGQA(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        self.embedding = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.final_norm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, past=None, use_cache=False):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError('Expected nonempty [batch, sequence] token IDs')
        if past is not None:
            if self.training or not use_cache:
                raise ValueError('Cache is an evaluation-only feature')
            if len(past) != len(self.layers):
                raise ValueError('Expected one cache entry per layer')
            lengths = {kv[0].shape[-2] for kv in past}
            if len(lengths) != 1:
                raise ValueError('Cache lengths disagree')
        offset = 0 if past is None else past[0][0].shape[-2]
        if offset + ids.shape[1] > self.config.max_seq_len:
            raise ValueError('Context budget exceeded; start a new sequence')
        x = self.embedding(ids)
        present = []
        for index, layer in enumerate(self.layers):
            x, kv = layer(x, None if past is None else past[index], use_cache)
            present.append(kv)
        return self.head(self.final_norm(x)), (present if use_cache else None)

    @torch.no_grad()
    def generate(self, ids, new_tokens=80, cached=True, temperature=0.0):
        if new_tokens < 0 or temperature < 0:
            raise ValueError('Generation arguments must be nonnegative')
        if ids.shape[1] + new_tokens > self.config.max_seq_len:
            raise ValueError('Prompt plus generation exceeds context budget')
        self.eval()
        result, past, current = ids.clone(), None, ids
        for _ in range(new_tokens):
            logits, past = self(current if cached else result,
                                 past=past if cached else None, use_cache=cached)
            scores = logits[:, -1]
            token = (scores.argmax(-1, keepdim=True) if temperature == 0 else
                     torch.multinomial((scores / temperature).softmax(-1), 1))
            result = torch.cat((result, token), dim=1)
            current = token
        return result


def cache_bytes(cache):
    """Logical bytes in compact K/V tensors, not peak allocated memory."""
    return sum(t.numel() * t.element_size() for pair in cache for t in pair)
