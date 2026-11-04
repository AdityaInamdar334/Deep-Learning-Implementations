import math
import torch
from torch import nn
from torch.nn import functional as F

class LoRAProjection(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.0,
                 transposed=False):
        super().__init__()
        if rank < 1:
            raise ValueError('rank must be positive')
        self.base = base
        self.base.requires_grad_(False)
        self.transposed = transposed
        if transposed:
            in_features, out_features = base.weight.shape
        else:
            out_features, in_features = base.weight.shape
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(base.weight.new_empty(rank, in_features))
        self.B = nn.Parameter(base.weight.new_zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        correction = F.linear(F.linear(self.dropout(x), self.A), self.B)
        return self.base(x) + correction * self.scale

    @torch.no_grad()
    def merged_copy(self):
        import copy
        if self.training:
            raise RuntimeError('Call eval() before merging')
        result = copy.deepcopy(self.base)
        delta = (self.B @ self.A) * self.scale
        if self.transposed:
            delta = delta.T
        result.weight.add_(delta)
        return result


class LoRAEmbedding(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if rank < 1:
            raise ValueError('rank must be positive')
        if base.padding_idx is not None or base.max_norm is not None:
            raise ValueError('This teaching layer excludes padding/max_norm')
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.codes = nn.Parameter(
            base.weight.new_empty(base.num_embeddings, rank))
        self.up = nn.Parameter(
            base.weight.new_zeros(base.embedding_dim, rank))
        nn.init.normal_(self.codes)

    def forward(self, token_ids):
        low = F.embedding(token_ids, self.codes)
        return self.base(token_ids) + F.linear(low, self.up) * self.scale
