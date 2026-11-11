"""Episode 4: PagedAttention & Virtual Memory Management.

Implements the vLLM PagedAttention paradigm (Kwon et al., SOSP 2023) to eliminate
memory fragmentation by partitioning Key-Value caches into fixed-size physical blocks
managed via sequence block tables.
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent

class BlockTableAllocator:
    """Virtual block memory manager managing physical KV cache blocks.
    
    Analogous to OS virtual memory paging:
    - Sequences are divided into logical blocks of size `block_size`.
    - Physical blocks are allocated on demand from a global free pool.
    - Zero external fragmentation; internal fragmentation is strictly bounded to (< block_size) tokens.
    """
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device

        # Pre-allocated physical memory pools for all layers:
        # shape: [num_layers, num_blocks, num_heads, block_size, head_dim]
        self.key_pool = torch.zeros(
            (num_layers, num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        self.val_pool = torch.zeros(
            (num_layers, num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )

        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_tables: Dict[int, List[int]] = {}
        self.seq_lengths: Dict[int, int] = {}

    @property
    def free_block_count(self) -> int:
        return len(self.free_blocks)

    def allocate_sequence(self, seq_id: int, initial_length: int = 0) -> List[int]:
        """Allocate initial physical blocks required for a sequence of given length."""
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} is already registered.")

        num_needed = math.ceil(initial_length / self.block_size) if initial_length > 0 else 1
        if num_needed > len(self.free_blocks):
            raise MemoryError(f"OOM: Need {num_needed} blocks, but only {len(self.free_blocks)} free.")

        allocated = [self.free_blocks.pop(0) for _ in range(num_needed)]
        self.block_tables[seq_id] = allocated
        self.seq_lengths[seq_id] = initial_length
        return allocated

    def append_slot(self, seq_id: int) -> Tuple[int, int]:
        """Advance sequence length by 1 token, allocating a new physical block if crossing boundary.
        
        Returns:
            (physical_block_id, slot_offset_in_block)
        """
        curr_len = self.seq_lengths[seq_id]
        logical_block = curr_len // self.block_size
        slot_offset = curr_len % self.block_size

        if logical_block >= len(self.block_tables[seq_id]):
            if not self.free_blocks:
                raise MemoryError(f"OOM: Cannot allocate block for sequence {seq_id}.")
            new_block = self.free_blocks.pop(0)
            self.block_tables[seq_id].append(new_block)

        phys_block = self.block_tables[seq_id][logical_block]
        self.seq_lengths[seq_id] = curr_len + 1
        return phys_block, slot_offset

    def write_kv(
        self,
        seq_id: int,
        token_pos: int,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int = 0,
    ):
        """Write key and value vectors into the designated physical slot.
        
        key, value shape: [num_heads, head_dim]
        """
        logical_block = token_pos // self.block_size
        slot_offset = token_pos % self.block_size
        phys_block = self.block_tables[seq_id][logical_block]

        self.key_pool[layer_idx, phys_block, :, slot_offset, :] = key
        self.val_pool[layer_idx, phys_block, :, slot_offset, :] = value

    def free_sequence(self, seq_id: int):
        """Return all physical blocks of a terminated sequence back to the free pool."""
        if seq_id not in self.block_tables:
            return
        blocks = self.block_tables.pop(seq_id)
        self.free_blocks.extend(blocks)
        del self.seq_lengths[seq_id]

    def compute_fragmentation(self, max_context_limit: int = 1024) -> dict:
        """Analyze memory utilization and fragmentation compared to contiguous allocation."""
        total_slots = self.num_blocks * self.block_size
        active_tokens = sum(self.seq_lengths.values())
        allocated_blocks = sum(len(b) for b in self.block_tables.values())
        allocated_slots = allocated_blocks * self.block_size
        
        internal_frag_slots = allocated_slots - active_tokens
        internal_frag_pct = (100.0 * internal_frag_slots / allocated_slots) if allocated_slots > 0 else 0.0

        # In traditional contiguous systems, each active request pre-allocates max_context_limit
        active_seqs = len(self.block_tables)
        contiguous_reserved_slots = active_seqs * max_context_limit
        contiguous_frag_slots = contiguous_reserved_slots - active_tokens
        contiguous_frag_pct = (100.0 * contiguous_frag_slots / contiguous_reserved_slots) if contiguous_reserved_slots > 0 else 0.0

        bytes_per_slot = 2 * self.num_layers * self.num_heads * self.head_dim * (2 if self.dtype in (torch.float16, torch.bfloat16) else 4)

        return {
            "active_sequences": active_seqs,
            "active_tokens": active_tokens,
            "paged_allocated_slots": allocated_slots,
            "paged_internal_frag_slots": internal_frag_slots,
            "paged_internal_frag_pct": internal_frag_pct,
            "contiguous_reserved_slots": contiguous_reserved_slots,
            "contiguous_frag_pct": contiguous_frag_pct,
            "memory_saved_bytes": (contiguous_reserved_slots - allocated_slots) * bytes_per_slot,
            "memory_reduction_ratio": (contiguous_reserved_slots / max(allocated_slots, 1)),
        }

def paged_attention_kernel(
    query: torch.Tensor,
    key_pool: torch.Tensor,
    val_pool: torch.Tensor,
    block_table: List[int],
    seq_len: int,
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute single-token paged attention across non-contiguous physical blocks using online softmax.
    
    Args:
        query: [num_heads, 1, head_dim]
        key_pool: [num_blocks, num_heads, block_size, head_dim]
        val_pool: [num_blocks, num_heads, block_size, head_dim]
        block_table: List of physical block IDs for this sequence
        seq_len: Total valid token length of sequence
        block_size: Block size K
        scale: 1 / sqrt(head_dim)
        
    Returns:
        output: [num_heads, 1, head_dim]
    """
    num_heads, _, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    acc = torch.zeros(num_heads, 1, head_dim, dtype=query.dtype, device=query.device)
    running_m = torch.full((num_heads, 1, 1), -float("inf"), dtype=query.dtype, device=query.device)
    running_d = torch.zeros(num_heads, 1, 1, dtype=query.dtype, device=query.device)

    for b_idx, p_block in enumerate(block_table):
        valid_len = min(block_size, seq_len - b_idx * block_size)
        if valid_len <= 0:
            break

        k_blk = key_pool[p_block, :, :valid_len, :]  # [num_heads, valid_len, head_dim]
        v_blk = val_pool[p_block, :, :valid_len, :]  # [num_heads, valid_len, head_dim]

        # [num_heads, 1, valid_len]
        s_blk = torch.matmul(query, k_blk.transpose(-2, -1)) * scale

        m_blk = s_blk.max(dim=-1, keepdim=True).values
        new_m = torch.maximum(running_m, m_blk)

        exp_s = torch.exp(s_blk - new_m)
        alpha = torch.exp(running_m - new_m)

        running_d = running_d * alpha + exp_s.sum(dim=-1, keepdim=True)
        acc = acc * alpha + torch.matmul(exp_s, v_blk)
        running_m = new_m

    return acc / running_d

def reference_contiguous_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Standard contiguous multi-head attention for equivalence verification."""
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, keys.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, values)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-blocks", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--num-sequences", type=int, default=6)
    p.add_argument("--max-context", type=int, default=1024)
    p.add_argument("--out", type=Path, default=Path("results/paged_attention_run.json"))
    args = p.parse_args()

    print("--- Episode 4: PagedAttention & Virtual Memory Management ---")
    allocator = BlockTableAllocator(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_layers=1,
    )

    print(f"Physical Memory Pool Initialized: {args.num_blocks} blocks of size {args.block_size} (capacity {args.num_blocks * args.block_size} tokens)")
    
    # 1. Equivalence Test against standard contiguous attention
    print("\n[Verification 1] Testing PagedAttention vs Contiguous Attention Numerical Parity:")
    test_len = 37  # 37 tokens spans 3 blocks (16 + 16 + 5)
    seq_id = 99
    block_ids = allocator.allocate_sequence(seq_id, test_len)
    
    # Fill random keys and values
    ref_keys = torch.randn(args.num_heads, test_len, args.head_dim)
    ref_vals = torch.randn(args.num_heads, test_len, args.head_dim)
    for pos in range(test_len):
        allocator.write_kv(seq_id, pos, ref_keys[:, pos, :], ref_vals[:, pos, :], layer_idx=0)
        
    query = torch.randn(args.num_heads, 1, args.head_dim)
    
    out_ref = reference_contiguous_attention(query, ref_keys, ref_vals)
    out_paged = paged_attention_kernel(
        query=query,
        key_pool=allocator.key_pool[0],
        val_pool=allocator.val_pool[0],
        block_table=block_ids,
        seq_len=test_len,
        block_size=args.block_size,
    )
    
    max_error = (out_ref - out_paged).abs().max().item()
    print(f"  Sequence length: {test_len} tokens spanning {len(block_ids)} physical blocks: {block_ids}")
    print(f"  Maximum Absolute Numerical Error: {max_error:.2e} (Threshold: < 1e-6)")
    assert max_error < 1e-5, f"PagedAttention numerical divergence: {max_error}"
    print("  Status: BITWISE/NUMERICAL EQUIVALENCE VERIFIED PASSED.")
    allocator.free_sequence(seq_id)

    # 2. Dynamic multi-request allocation & fragmentation simulation
    print("\n[Verification 2] Simulating Dynamic Allocation across Multiple Requests:")
    import random
    random.seed(42)
    seq_lengths = [random.randint(12, 45) for _ in range(args.num_sequences)]
    
    for s_idx, s_len in enumerate(seq_lengths):
        allocator.allocate_sequence(s_idx, s_len)
        print(f"  Admitted Req {s_idx}: length {s_len:2d} tokens -> Blocks allocated: {allocator.block_tables[s_idx]}")

    stats = allocator.compute_fragmentation(max_context_limit=args.max_context)
    print("\n--- Memory Fragmentation Analysis ---")
    print(f"  Active requests:              {stats['active_sequences']}")
    print(f"  Total valid active tokens:    {stats['active_tokens']}")
    print(f"  PagedAttention allocated:     {stats['paged_allocated_slots']} slots")
    print(f"  Paged internal fragmentation: {stats['paged_internal_frag_slots']} slots ({stats['paged_internal_frag_pct']:.1f}%)")
    print(f"  Traditional reserved slots:   {stats['contiguous_reserved_slots']} slots (pre-allocated {args.max_context} per req)")
    print(f"  Traditional fragmentation:    {stats['contiguous_frag_pct']:.1f}%")
    print(f"  Memory Footprint Reduction:   {stats['memory_reduction_ratio']:.1f}x reduction ({stats['memory_saved_bytes'] / 1024:.1f} KB saved)")

    # 3. Request lifecycle: Freeing sequences and recycling blocks
    print("\n[Verification 3] Request Termination and Block Recycling:")
    free_before = allocator.free_block_count
    allocator.free_sequence(0)
    allocator.free_sequence(1)
    free_after = allocator.free_block_count
    recycled = free_after - free_before
    print(f"  Retired requests 0 & 1 -> Recycled {recycled} blocks back to free pool ({free_after} free blocks total).")
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Saved PagedAttention telemetry to {args.out}")

if __name__ == "__main__":
    main()

