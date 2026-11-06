"""MoE Capacity Planner & Distributed Serving Simulator.

Models weight memory footprints across precisions (FP16, FP8, FP4),
KV-cache growth under MHA, GQA, and Multi-Head Latent Attention (MLA),
hardware node capacity thresholds (H100 80GB, H200 141GB), and
All-to-All network communication latency under Expert Parallelism.
"""

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MoEModelConfig:
    name: str
    total_params_b: float       # Total parameters in billions
    active_params_b: float      # Active parameters per token in billions
    num_layers: int
    num_routed_experts: int
    top_k: int
    hidden_dim: int = 8192
    kv_heads: int = 8
    head_dim: int = 128
    attention_type: str = "MLA"  # "MHA", "GQA", "MLA", "Sparse"
    mla_latent_dim: int = 512    # Latent dimension for Multi-head Latent Attention
    sparse_attention_ratio: float = 0.15


@dataclass
class HardwareNodeConfig:
    name: str
    gpu_memory_gb: float
    gpus_per_node: int
    interconnect_bandwidth_gbps: float  # e.g., NVLink 900 GB/s per GPU, InfiniBand 400 Gbps inter-node


FRONTIER_MODELS: Dict[str, MoEModelConfig] = {
    "Kimi K3.1 Pro": MoEModelConfig(
        name="Kimi K3.1 Pro",
        total_params_b=2800.0,
        active_params_b=104.0,
        num_layers=93,
        num_routed_experts=896,
        top_k=16,
        hidden_dim=12288,
        kv_heads=8,
        head_dim=128,
        attention_type="Sparse",
        sparse_attention_ratio=0.18,
    ),
    "Mistral Large 4": MoEModelConfig(
        name="Mistral Large 4",
        total_params_b=1050.0,
        active_params_b=52.0,
        num_layers=72,
        num_routed_experts=128,
        top_k=4,
        hidden_dim=8192,
        kv_heads=8,
        head_dim=128,
        attention_type="MLA",
        mla_latent_dim=512,
    ),
    "Tencent Hy4": MoEModelConfig(
        name="Tencent Hy4",
        total_params_b=770.0,
        active_params_b=49.0,
        num_layers=64,
        num_routed_experts=64,
        top_k=4,
        hidden_dim=8192,
        kv_heads=8,
        head_dim=128,
        attention_type="Sparse",
        sparse_attention_ratio=0.20,
    ),
    "Inkling 2 Small": MoEModelConfig(
        name="Inkling 2 Small",
        total_params_b=594.0,
        active_params_b=17.0,
        num_layers=48,
        num_routed_experts=384,
        top_k=8,
        hidden_dim=6144,
        kv_heads=8,
        head_dim=128,
        attention_type="GQA",
    ),
}

HARDWARE_H100 = HardwareNodeConfig(
    name="NVIDIA H100 SXM5",
    gpu_memory_gb=80.0,
    gpus_per_node=8,
    interconnect_bandwidth_gbps=900.0,
)

HARDWARE_H200 = HardwareNodeConfig(
    name="NVIDIA H200 SXM5",
    gpu_memory_gb=141.0,
    gpus_per_node=8,
    interconnect_bandwidth_gbps=900.0,
)


def estimate_weight_memory(total_params_b: float, bits: int, include_metadata: bool = True) -> float:
    """Calculates theoretical and realistic quantized weight memory footprint in GB."""
    raw_gb = (total_params_b * 1e9 * (bits / 8)) / (1024**3)
    if include_metadata and bits <= 8:
        # Quantization scales, zero-points, and block codebook overhead (~3.5% overhead)
        return raw_gb * 1.035
    return raw_gb


def estimate_kv_cache_memory(
    model: MoEModelConfig,
    context_tokens: int,
    batch_size: int = 1,
    precision_bytes: int = 2,
) -> float:
    """Calculates KV cache footprint in GB across attention architectures:

    - MHA: 2 * layers * tokens * heads * head_dim * bytes
    - GQA: 2 * layers * tokens * kv_heads * head_dim * bytes
    - MLA: layers * tokens * (latent_dim + rope_dim) * bytes
    - Sparse: scales by sparse_attention_ratio
    """
    if model.attention_type == "MLA":
        # Multi-Head Latent Attention compresses KV representations into latent vector
        rope_dim = 64
        per_token_bytes = (model.mla_latent_dim + rope_dim) * precision_bytes
        total_bytes = batch_size * model.num_layers * context_tokens * per_token_bytes
    elif model.attention_type == "Sparse":
        # Sparse attention indexes only a fraction of historical tokens
        active_tokens = context_tokens * model.sparse_attention_ratio
        per_token_bytes = 2 * model.kv_heads * model.head_dim * precision_bytes
        total_bytes = batch_size * model.num_layers * active_tokens * per_token_bytes
    else:  # GQA or standard MHA
        per_token_bytes = 2 * model.kv_heads * model.head_dim * precision_bytes
        total_bytes = batch_size * model.num_layers * context_tokens * per_token_bytes

    return total_bytes / (1024**3)


def calculate_serving_capacity(
    model: MoEModelConfig,
    precision_bits: int,
    context_tokens: int,
    hardware: HardwareNodeConfig,
    batch_size: int = 1,
    workspace_overhead_gb: float = 12.0,
) -> Dict[str, Any]:
    """Computes total GPU memory requirements and required GPU/node counts."""
    weight_mem = estimate_weight_memory(model.total_params_b, precision_bits)
    kv_mem = estimate_kv_cache_memory(model, context_tokens, batch_size)
    total_mem = weight_mem + kv_mem + (workspace_overhead_gb * ceil(weight_mem / hardware.gpu_memory_gb))

    min_gpus = ceil(total_mem / hardware.gpu_memory_gb)
    min_nodes = ceil(min_gpus / hardware.gpus_per_node)
    total_allocated_mem = min_nodes * hardware.gpus_per_node * hardware.gpu_memory_gb
    headroom_gb = total_allocated_mem - total_mem

    return {
        "model": model.name,
        "precision_bits": precision_bits,
        "context_tokens": context_tokens,
        "weight_memory_gb": round(weight_mem, 1),
        "kv_cache_memory_gb": round(kv_mem, 1),
        "total_memory_needed_gb": round(total_mem, 1),
        "hardware": hardware.name,
        "gpus_required": min_gpus,
        "nodes_required": min_nodes,
        "total_vram_allocated_gb": round(total_allocated_mem, 1),
        "headroom_gb": round(headroom_gb, 1),
    }


def estimate_all_to_all_dispatch_latency(
    active_params_b: float,
    top_k: int,
    hidden_dim: int,
    num_experts: int,
    gpus_count: int,
    interconnect_bandwidth_gbps: float = 900.0,
    batch_tokens: int = 2048,
) -> float:
    """Estimates All-to-All token dispatch latency across expert-parallel GPUs in milliseconds.

    MoE routing requires dispatching tokens to expert-holding devices:
    Transfer Bytes = batch_tokens * top_k * hidden_dim * 2 bytes * (1 - 1/gpus)
    """
    if gpus_count <= 1:
        return 0.0

    transfer_bytes = batch_tokens * top_k * hidden_dim * 2 * (1.0 - (1.0 / gpus_count))
    transfer_gb = transfer_bytes / (1024**3)
    latency_sec = transfer_gb / interconnect_bandwidth_gbps
    return latency_sec * 1000.0  # in ms
