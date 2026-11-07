"""Unit tests for the MoE Capacity Planner & Benchmark Version Analyzer."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moe_capacity_planner import (
    FRONTIER_MODELS,
    HARDWARE_H100,
    estimate_weight_memory,
    estimate_kv_cache_memory,
    calculate_serving_capacity,
    estimate_all_to_all_dispatch_latency,
)
from benchmark_version_analyzer import analyze_version_gap, compute_task_economics


class TestMoECapacity(unittest.TestCase):

    def test_weight_memory_scaling(self):
        """FP4 memory must be approximately half of FP8 and a quarter of FP16."""
        params_b = 1000.0
        mem_fp16 = estimate_weight_memory(params_b, 16, include_metadata=False)
        mem_fp8 = estimate_weight_memory(params_b, 8, include_metadata=False)
        mem_fp4 = estimate_weight_memory(params_b, 4, include_metadata=False)

        self.assertAlmostEqual(mem_fp8, mem_fp16 / 2.0, places=1)
        self.assertAlmostEqual(mem_fp4, mem_fp16 / 4.0, places=1)

    def test_kv_cache_mla_compression(self):
        """Multi-head Latent Attention (MLA) must consume less memory than uncompressed MHA."""
        mistral = FRONTIER_MODELS["Mistral Large 4"]
        context_tokens = 131_072

        kv_mla = estimate_kv_cache_memory(mistral, context_tokens)

        # Create hypothetical uncompressed MHA version of Mistral
        import copy
        mistral_mha = copy.deepcopy(mistral)
        mistral_mha.attention_type = "MHA"
        mistral_mha.kv_heads = 64  # Full 64 heads rather than latent compression
        kv_mha = estimate_kv_cache_memory(mistral_mha, context_tokens)

        self.assertLess(kv_mla, kv_mha, "MLA compressed KV-cache must be smaller than full MHA KV-cache")

    def test_capacity_plan_node_rounding(self):
        """GPU count and node count must correctly round up to integer multiples."""
        model = FRONTIER_MODELS["Inkling 2 Small"]
        plan = calculate_serving_capacity(model, precision_bits=4, context_tokens=8192, hardware=HARDWARE_H100)

        self.assertGreaterEqual(plan["gpus_required"], 4)
        self.assertGreaterEqual(plan["nodes_required"], 1)
        self.assertEqual(plan["total_vram_allocated_gb"], plan["nodes_required"] * 8 * 80.0)

    def test_cross_version_gap_detection(self):
        """Comparing v4.3 and v5 scores must detect the manufactured cross-version inflation."""
        res = analyze_version_gap("Kimi K3.1 Pro", "Mistral Large 4")
        self.assertEqual(res["true_v5_same_version_gap"], 20.0)
        self.assertEqual(res["manufactured_cross_version_gap"], 29.0)
        self.assertEqual(res["fallacy_inflation_points"], 9.0)

    def test_all_to_all_dispatch_latency(self):
        """All-to-All latency must be 0 on single GPU and strictly positive across a cluster."""
        lat_1 = estimate_all_to_all_dispatch_latency(52.0, 4, 8192, 128, gpus_count=1)
        lat_8 = estimate_all_to_all_dispatch_latency(52.0, 4, 8192, 128, gpus_count=8)

        self.assertEqual(lat_1, 0.0)
        self.assertGreater(lat_8, 0.0)

    def test_task_economics_verbosity(self):
        """Higher token generation count must result in longer runtime and higher compute cost."""
        eco_concise = compute_task_economics(active_params_b=50.0, tokens_generated=110_000)
        eco_verbose = compute_task_economics(active_params_b=50.0, tokens_generated=154_000)

        self.assertGreater(eco_verbose["task_duration_seconds"], eco_concise["task_duration_seconds"])
        self.assertGreater(eco_verbose["compute_cost_usd"], eco_concise["compute_cost_usd"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

