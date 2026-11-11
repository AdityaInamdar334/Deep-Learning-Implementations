"""Episode 5: Continuous (Iteration-Level) Batching Scheduler.

Eliminates head-of-line blocking and GPU idle time by dynamically admitting newly
arrived requests into active decoding iterations as soon as existing requests
reach EOS or finish generation (Orca / vLLM iteration-level scheduling).
"""
import argparse
import enum
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent

class RequestState(enum.Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"

@dataclass
class Request:
    req_id: int
    prompt_text: str
    prompt_ids: List[int]
    max_new_tokens: int
    arrival_step: int
    state: RequestState = RequestState.WAITING
    start_step: Optional[int] = None
    finish_step: Optional[int] = None
    generated_tokens: List[int] = field(default_factory=list)
    past_key_values: Any = None
    ttft_ms: Optional[float] = None
    total_ms: Optional[float] = None
    queue_wait_steps: int = 0
    total_service_steps: int = 0

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

class ContinuousBatchEngine:
    """Iteration-level continuous batching inference engine.
    
    Operates at per-iteration granularity:
    1. Check and retire completed requests immediately.
    2. Admit waiting requests up to max_batch_size capacity.
    3. Execute iteration forward passes (prefill for new, decode for running).
    4. Release resources upon request completion without stalling peers.
    """
    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.device = device or next(model.parameters()).device
        self.eos_token_id = tokenizer.eos_token_id

        self.waiting_queue: List[Request] = []
        self.active_pool: List[Request] = []
        self.completed_requests: List[Request] = []
        self.step_idx: int = 0
        self.step_events: List[dict] = []

    def submit_request(self, request: Request):
        self.waiting_queue.append(request)

    @torch.inference_mode()
    def step(self) -> bool:
        """Executes a single continuous batching iteration.
        
        Returns:
            True if work was performed, False if all queues are empty.
        """
        # 1. Admit new requests up to available capacity
        slots_available = self.max_batch_size - len(self.active_pool)
        admitted_this_step = []
        
        while slots_available > 0 and self.waiting_queue:
            # Check if request has arrived by current step
            candidate = self.waiting_queue[0]
            if candidate.arrival_step <= self.step_idx:
                req = self.waiting_queue.pop(0)
                req.state = RequestState.RUNNING
                req.start_step = self.step_idx
                req.queue_wait_steps = self.step_idx - req.arrival_step
                self.active_pool.append(req)
                admitted_this_step.append(req.req_id)
                slots_available -= 1
            else:
                break

        if not self.active_pool and not self.waiting_queue:
            return False

        if not self.active_pool:
            # Waiting for scheduled future arrivals
            self.step_idx += 1
            return True

        # 2. Execute forward passes for all active requests
        newly_completed = []
        step_start_time = time.perf_counter_ns()

        for req in list(self.active_pool):
            is_prefill = (len(req.generated_tokens) == 0)
            t0 = time.perf_counter_ns()

            if is_prefill:
                # Prefill pass: process prompt tokens
                input_tensor = torch.tensor([req.prompt_ids], dtype=torch.long, device=self.device)
                out = self.model(input_ids=input_tensor, use_cache=True)
                req.past_key_values = out.past_key_values
                token = out.logits[:, -1, :].argmax(dim=-1).item()
                synchronize(self.device)
                t1 = time.perf_counter_ns()
                req.ttft_ms = (t1 - t0) / 1e6
            else:
                # Incremental decode pass: process single token
                last_token = torch.tensor([[req.generated_tokens[-1]]], dtype=torch.long, device=self.device)
                out = self.model(input_ids=last_token, past_key_values=req.past_key_values, use_cache=True)
                req.past_key_values = out.past_key_values
                token = out.logits[:, -1, :].argmax(dim=-1).item()
                synchronize(self.device)

            req.generated_tokens.append(token)
            req.total_service_steps += 1

            # Check termination criteria (EOS or reached max_new_tokens)
            hit_eos = (self.eos_token_id is not None and token == self.eos_token_id)
            hit_limit = (len(req.generated_tokens) >= req.max_new_tokens)

            if hit_eos or hit_limit:
                req.state = RequestState.COMPLETED
                req.finish_step = self.step_idx
                req.past_key_values = None  # Release memory immediately!
                self.active_pool.remove(req)
                self.completed_requests.append(req)
                newly_completed.append(req.req_id)

        step_duration_ms = (time.perf_counter_ns() - step_start_time) / 1e6

        self.step_events.append({
            "step": self.step_idx,
            "active_count": len(self.active_pool) + len(newly_completed),
            "admitted_reqs": admitted_this_step,
            "completed_reqs": newly_completed,
            "step_duration_ms": step_duration_ms,
        })

        self.step_idx += 1
        return True

    def run_all(self, max_steps: int = 1000) -> List[Request]:
        """Run engine until all submitted requests finish."""
        for _ in range(max_steps):
            active = self.step()
            if not active and not self.waiting_queue:
                break
        return self.completed_requests

class StaticBatchEngineSimulator:
    """Simulates static batching where the engine waits for the entire batch to finish."""
    def __init__(self, model, tokenizer, batch_size: int = 4, device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device or next(model.parameters()).device
        self.eos_token_id = tokenizer.eos_token_id

    def run_workload(self, requests: List[Request]) -> List[Request]:
        waiting = list(requests)
        completed = []
        current_step = 0

        while waiting:
            # Collect available requests up to batch_size
            ready = [r for r in waiting if r.arrival_step <= current_step]
            if not ready:
                current_step = min(r.arrival_step for r in waiting)
                ready = [r for r in waiting if r.arrival_step <= current_step]

            batch = ready[:self.batch_size]
            for r in batch:
                waiting.remove(r)
                r.start_step = current_step
                r.queue_wait_steps = current_step - r.arrival_step

            # In static batching, all requests run until the longest request completes
            batch_tokens_needed = [r.max_new_tokens for r in batch]
            active_flags = [True] * len(batch)
            step_in_batch = 0

            # Step 0: Prefills
            for r in batch:
                inp = torch.tensor([r.prompt_ids], dtype=torch.long, device=self.device)
                out = self.model(inp, use_cache=True)
                r.past_key_values = out.past_key_values
                tok = out.logits[:, -1, :].argmax(dim=-1).item()
                r.generated_tokens.append(tok)
                if self.eos_token_id is not None and tok == self.eos_token_id:
                    active_flags[batch.index(r)] = False

            step_in_batch += 1
            current_step += 1

            # Decode steps: must continue until all reach EOS or max tokens
            while any(active_flags):
                for i, r in enumerate(batch):
                    if active_flags[i]:
                        last_tok = torch.tensor([[r.generated_tokens[-1]]], dtype=torch.long, device=self.device)
                        out = self.model(last_tok, past_key_values=r.past_key_values, use_cache=True)
                        r.past_key_values = out.past_key_values
                        tok = out.logits[:, -1, :].argmax(dim=-1).item()
                        r.generated_tokens.append(tok)
                        if (self.eos_token_id is not None and tok == self.eos_token_id) or (len(r.generated_tokens) >= r.max_new_tokens):
                            active_flags[i] = False

                step_in_batch += 1
                current_step += 1

            # All finished: batch completes together!
            for r in batch:
                r.finish_step = current_step
                r.state = RequestState.COMPLETED
                r.past_key_values = None
                completed.append(r)

        return completed

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "fixture")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--out", type=Path, default=Path("results/continuous_batching_run.json"))
    args = p.parse_args()

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    print("--- Episode 5: Continuous (Iteration-Level) Batching ---")
    print(f"Hardware: {device.type} | Max Concurrent Active Slots: {args.batch_size}\n")

    # Define a realistic arrival stream with staggered arrivals and heterogeneous generation targets
    workload_specs = [
        # (id, prompt, max_tokens, arrival_step)
        (0, "sensor reports hot", 6, 0),             # Arrives step 0, short prompt
        (1, "sensor reports cold . fan off .", 2, 0),  # Arrives step 0, emits EOS quickly (step 1)
        (2, "fan on", 8, 0),                          # Arrives step 0, longer decode
        (3, "sensor reports hot . fan on .", 2, 1),   # Arrives step 1, short
        (4, "sensor reports cold", 7, 2),             # Arrives step 2
        (5, "sensor reports cold .", 5, 3),           # Arrives step 3
    ]

    # --- Run Continuous Batching ---
    cb_engine = ContinuousBatchEngine(model, tokenizer, max_batch_size=args.batch_size, device=device)
    cb_requests = [
        Request(
            req_id=spec[0],
            prompt_text=spec[1],
            prompt_ids=tokenizer.encode(spec[1], add_special_tokens=False),
            max_new_tokens=spec[2],
            arrival_step=spec[3],
        )
        for spec in workload_specs
    ]
    for r in cb_requests:
        cb_engine.submit_request(r)

    print("Executing Continuous Batching Engine (Iteration-Level Scheduling):")
    cb_engine.run_all()

    print("\nIteration Trace:")
    for ev in cb_engine.step_events:
        adm = f"+Reqs {ev['admitted_reqs']}" if ev['admitted_reqs'] else ""
        comp = f"-Done {ev['completed_reqs']}" if ev['completed_reqs'] else ""
        print(f"  Step {ev['step']:2d} | Active: {ev['active_count']} | {adm:12s} | {comp:12s} | {ev['step_duration_ms']:.2f} ms")

    # --- Run Static Batching on identical workload for comparison ---
    sb_engine = StaticBatchEngineSimulator(model, tokenizer, batch_size=args.batch_size, device=device)
    sb_requests = [
        Request(
            req_id=spec[0],
            prompt_text=spec[1],
            prompt_ids=tokenizer.encode(spec[1], add_special_tokens=False),
            max_new_tokens=spec[2],
            arrival_step=spec[3],
        )
        for spec in workload_specs
    ]
    sb_completed = sb_engine.run_workload(sb_requests)

    # --- Performance Comparison Summary ---
    cb_req_map = {r.req_id: r for r in cb_engine.completed_requests}
    sb_req_map = {r.req_id: r for r in sb_completed}

    print("\n--- Comparative Metrics: Continuous vs Static Batching ---")
    print(f"{'Req ID':<8}{'Arrival':<10}{'CB Wait':<10}{'CB Finish':<12}{'SB Wait':<10}{'SB Finish':<12}{'Latency Reduction'}")
    print("-" * 75)

    cb_waits = []
    sb_waits = []
    cb_latencies = []
    sb_latencies = []

    for spec in workload_specs:
        rid = spec[0]
        cb_r = cb_req_map[rid]
        sb_r = sb_req_map[rid]
        
        cb_latency = cb_r.finish_step - cb_r.arrival_step
        sb_latency = sb_r.finish_step - sb_r.arrival_step

        cb_waits.append(cb_r.queue_wait_steps)
        sb_waits.append(sb_r.queue_wait_steps)
        cb_latencies.append(cb_latency)
        sb_latencies.append(sb_latency)

        diff_steps = sb_latency - cb_latency
        print(f"Req {rid:<4}{spec[3]:<10}{cb_r.queue_wait_steps:<10}{cb_r.finish_step:<12}{sb_r.queue_wait_steps:<10}{sb_r.finish_step:<12}{diff_steps:+d} steps faster")

    cb_makespan = max(r.finish_step for r in cb_engine.completed_requests)
    sb_makespan = max(r.finish_step for r in sb_completed)

    print("-" * 75)
    print(f"Average Queue Wait Steps: Continuous = {statistics.mean(cb_waits):.2f} steps | Static = {statistics.mean(sb_waits):.2f} steps")
    print(f"Average End-to-End Steps: Continuous = {statistics.mean(cb_latencies):.2f} steps | Static = {statistics.mean(sb_latencies):.2f} steps")
    print(f"Total Workload Makespan:  Continuous = {cb_makespan} steps | Static = {sb_makespan} steps ({(1 - cb_makespan/sb_makespan)*100:.1f}% faster)")

    summary = {
        "continuous_batching": {
            "makespan_steps": cb_makespan,
            "avg_queue_wait_steps": statistics.mean(cb_waits),
            "avg_latency_steps": statistics.mean(cb_latencies),
        },
        "static_batching": {
            "makespan_steps": sb_makespan,
            "avg_queue_wait_steps": statistics.mean(sb_waits),
            "avg_latency_steps": statistics.mean(sb_latencies),
        },
        "makespan_speedup_pct": (1 - cb_makespan / sb_makespan) * 100,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved Continuous Batching metrics to {args.out}")

if __name__ == "__main__":
    main()

