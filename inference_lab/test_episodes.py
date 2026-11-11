"""Unit and invariant test suite for Episodes 2, 3, 4, and 5 of the Inference Server."""
import math
import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import ROOT, generate as generate_uncached
from kv_cache import generate_with_cache, inspect_cache_memory
from batching import generate_batched, collate_left_padded
from paged_attention import BlockTableAllocator, paged_attention_kernel, reference_contiguous_attention
from continuous_batching import ContinuousBatchEngine, Request, RequestState

class InferenceEpisodesTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.device = torch.device("cpu")
        cls.model = AutoModelForCausalLM.from_pretrained(ROOT / "fixture").eval().to(cls.device)
        cls.tokenizer = AutoTokenizer.from_pretrained(ROOT / "fixture")

    # =========================================================================
    # Episode 2: Static KV Caching Invariants
    # =========================================================================
    def test_episode02_logit_parity_with_uncached_baseline(self):
        """Verify KV cached generation produces identical tokens to Episode 1 uncached generation."""
        prompt_tensor = torch.tensor([self.tokenizer.encode("sensor reports cold")], device=self.device)
        output_uncached, trace_uncached = generate_uncached(self.model, prompt_tensor, 6)
        output_cached, trace_cached = generate_with_cache(self.model, prompt_tensor, 6)

        self.assertTrue(torch.equal(output_uncached, output_cached))
        # Baseline recomputation processes 3 + 4 + 5 + 6 + 7 + 8 = 33 tokens
        self.assertEqual(trace_uncached["positions_processed"], 33)
        # KV cache processes prompt (3) + 1 token per decode step (5) = 8 tokens
        self.assertEqual(trace_cached["positions_processed"], 8)
        self.assertEqual(trace_cached["input_lengths"], [3, 1, 1, 1, 1, 1])

    def test_episode02_cache_inspection_and_termination(self):
        """Verify memory inspection and early EOS termination in KV cache."""
        prompt = torch.tensor([self.tokenizer.encode("sensor reports hot . fan on .")], device=self.device)
        output, trace = generate_with_cache(self.model, prompt, 8, eos=self.tokenizer.eos_token_id)
        self.assertEqual(output.tolist(), [[self.tokenizer.eos_token_id]])
        self.assertGreater(trace["cache_bytes"], 0)
        self.assertGreater(trace["cache_elements"], 0)

    # =========================================================================
    # Episode 3: Dynamic Request Batching Invariants
    # =========================================================================
    def test_episode03_batch_isolation_parity(self):
        """Verify item i in a left-padded batch produces exact same tokens as running alone."""
        p1 = self.tokenizer.encode("sensor reports hot", add_special_tokens=False) # len 3
        p2 = self.tokenizer.encode("sensor reports cold . fan off .", add_special_tokens=False) # len 7
        p3 = self.tokenizer.encode("fan on", add_special_tokens=False) # len 2

        # Run individually
        single_outs = []
        for p in [p1, p2, p3]:
            inp = torch.tensor([p], device=self.device)
            out, _ = generate_with_cache(self.model, inp, 6, eos=self.tokenizer.eos_token_id)
            single_outs.append(out.squeeze().tolist() if out.ndim > 1 else [out.item()])

        # Run in a single heterogeneous batch
        batched_outs, _ = generate_batched(
            self.model,
            [p1, p2, p3],
            max_new_tokens=6,
            eos_token_id=self.tokenizer.eos_token_id,
            device=self.device,
        )

        for i in range(3):
            # Normalize to list
            s = single_outs[i] if isinstance(single_outs[i], list) else [single_outs[i]]
            b = batched_outs[i]
            self.assertEqual(s, b, f"Mismatch on request {i}: single={s} vs batched={b}")

    def test_episode03_left_padding_collation_structure(self):
        """Verify left padding puts padding tokens strictly on the left side."""
        p1 = [10, 20]
        p2 = [30, 40, 50, 60]
        input_ids, attention_mask, pos_ids, lengths = collate_left_padded([p1, p2], pad_token_id=0, device=self.device)

        self.assertEqual(input_ids.shape, (2, 4))
        self.assertEqual(attention_mask.tolist(), [[0, 0, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(pos_ids[0].tolist(), [0, 0, 0, 1])
        self.assertEqual(pos_ids[1].tolist(), [0, 1, 2, 3])
        self.assertEqual(lengths, [2, 4])

    # =========================================================================
    # Episode 4: PagedAttention & Virtual Memory Invariants
    # =========================================================================
    def test_episode04_paged_attention_numerical_parity(self):
        """Verify paged online attention matches contiguous reference attention within 1e-5."""
        num_heads, head_dim, block_size, test_len = 4, 16, 8, 27
        allocator = BlockTableAllocator(
            num_blocks=16, block_size=block_size, num_heads=num_heads, head_dim=head_dim, num_layers=1
        )
        block_ids = allocator.allocate_sequence(1, test_len)
        self.assertEqual(len(block_ids), math.ceil(test_len / block_size))

        ref_k = torch.randn(num_heads, test_len, head_dim)
        ref_v = torch.randn(num_heads, test_len, head_dim)
        for pos in range(test_len):
            allocator.write_kv(1, pos, ref_k[:, pos, :], ref_v[:, pos, :], layer_idx=0)

        query = torch.randn(num_heads, 1, head_dim)
        out_ref = reference_contiguous_attention(query, ref_k, ref_v)
        out_paged = paged_attention_kernel(
            query=query,
            key_pool=allocator.key_pool[0],
            val_pool=allocator.val_pool[0],
            block_table=block_ids,
            seq_len=test_len,
            block_size=block_size,
        )

        max_diff = (out_ref - out_paged).abs().max().item()
        self.assertLess(max_diff, 1e-5)

    def test_episode04_block_allocation_and_deallocation(self):
        """Verify block allocator lifecycle, slot appending, and block recycling."""
        allocator = BlockTableAllocator(
            num_blocks=10, block_size=4, num_heads=2, head_dim=8, num_layers=1
        )
        self.assertEqual(allocator.free_block_count, 10)

        # Allocate sequence of 4 tokens (1 block)
        b = allocator.allocate_sequence(1, 4)
        self.assertEqual(len(b), 1)
        self.assertEqual(allocator.free_block_count, 9)

        # Append 1 token -> crosses boundary -> allocates second block
        p_block, slot = allocator.append_slot(1)
        self.assertEqual(slot, 0)
        self.assertEqual(len(allocator.block_tables[1]), 2)
        self.assertEqual(allocator.free_block_count, 8)

        # Free sequence -> returns 2 blocks
        allocator.free_sequence(1)
        self.assertEqual(allocator.free_block_count, 10)

    # =========================================================================
    # Episode 5: Continuous Batching Invariants
    # =========================================================================
    def test_episode05_early_retirement_and_dynamic_admission(self):
        """Verify that short requests retire immediately and new requests admit without head-of-line blocking."""
        engine = ContinuousBatchEngine(self.model, self.tokenizer, max_batch_size=2, device=self.device)
        
        # Req 0: "sensor reports hot . fan on ." emits EOS in 1 token
        req0 = Request(
            req_id=0,
            prompt_text="sensor reports hot . fan on .",
            prompt_ids=self.tokenizer.encode("sensor reports hot . fan on .", add_special_tokens=False),
            max_new_tokens=5,
            arrival_step=0,
        )
        # Req 1: generates 6 tokens
        req1 = Request(
            req_id=1,
            prompt_text="sensor reports cold",
            prompt_ids=self.tokenizer.encode("sensor reports cold", add_special_tokens=False),
            max_new_tokens=6,
            arrival_step=0,
        )
        # Req 2: waiting in queue, arrives step 0
        req2 = Request(
            req_id=2,
            prompt_text="sensor reports hot",
            prompt_ids=self.tokenizer.encode("sensor reports hot", add_special_tokens=False),
            max_new_tokens=4,
            arrival_step=0,
        )

        engine.submit_request(req0)
        engine.submit_request(req1)
        engine.submit_request(req2)

        # Step 0: Admits req0 and req1 (max_batch_size=2)
        engine.step()
        self.assertEqual(req0.state, RequestState.COMPLETED) # req0 finished immediately on step 0!
        self.assertEqual(req0.finish_step, 0)
        self.assertEqual(req1.state, RequestState.RUNNING)

        # Step 1: Slot vacated by req0 must admit req2 immediately!
        engine.step()
        self.assertEqual(req2.state, RequestState.RUNNING)
        self.assertEqual(req2.start_step, 1)

        # Run remaining
        engine.run_all()
        self.assertEqual(len(engine.completed_requests), 3)

if __name__ == "__main__":
    unittest.main()

