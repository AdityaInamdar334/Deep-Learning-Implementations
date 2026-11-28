# Mira Murati’s New AI Model Has 24 Layers. Every Token Goes Through 48.

*Inside Inkling 2 Mini, the looped transformer that separates model size from computational depth, and what it means for the future of local AI.*

On November 20, 2026, Thinking Machines Lab released a language model with 26.4 billion parameters.

Nothing particularly unusual about that number.

But the model has only 24 physical transformer layers, while each generated token passes through the equivalent of 48 layers.

The additional layers don’t come from additional weights. The model runs the middle 12 layers three times, repeatedly transforming its internal representation before deciding which token to generate.

Meanwhile, its mixture-of-experts architecture activates approximately 5.4 billion parameter instances per token when repeated computation is included, compared with the 27 billion parameters used by the dense Qwen3.8-27B model ([Thinking Machines Lab, 2026](https://thinkingmachines.ai/news/inkling-2-mini/); [Qwen Team, 2026](https://huggingface.co/Qwen/Qwen3.8-27B)).

This creates a fascinating distinction. Two language models can require approximately the same storage space, yet perform completely different amounts of computation and distribute that computation differently across their architectures.

For anyone running language models on a MacBook, building local coding agents, or researching more efficient transformer architectures, this raises an important question:

**Do we actually need more parameters to build more capable language models, or can we make better use of the parameters we already have?**

## 1. Three different ways to make a language model more capable

For much of the history of deep learning, improving a model meant increasing its capacity. We added layers, increased hidden dimensions, collected larger datasets, and trained networks using more computation.

However, modern language models give us at least three distinct ways to increase computational capacity.

A conventional dense transformer increases capacity by adding unique layers or making existing layers wider. A reasoning model can spend more computation generating intermediate reasoning tokens before producing its answer. A looped transformer introduces another mechanism: repeatedly applying the same learned transformation inside a forward computation.

These mechanisms are not mutually exclusive. Inkling itself is both a reasoning model and a looped transformer.

The important distinction is **where the additional computation occurs**.

A longer reasoning trace increases computation along the sequence dimension. A looped transformer increases computation along the depth dimension. This distinction becomes especially important on hardware where memory capacity and bandwidth are limited.

[Insert Figure 1: scaling_axes.png]

*Figure 1. Additional unique parameters, generated reasoning tokens, and recurrent computational depth are distinct ways of scaling computation.*

## 2. Inside Inkling 2 Mini: 24 physical layers, 48 computational layers

The architecture is best understood by dividing the model into three sections.

- **Prelude:** The first six transformer layers process the input.
- **Recurrent core:** The middle twelve transformer layers run three times using the same learned weights.
- **Coda:** The final six layers transform the resulting representation before output prediction.

[Insert Figure 2: architecture.png]

*Figure 2. Simplified Inkling 2 Mini architecture. The recurrent core represents repeated execution of the same twelve physical layers.*

Effective depth = 6 + 3 × 12 + 6 = 48 layers.

Physical depth = 6 + 12 + 6 = 24 layers.

The weights required for the extra computational steps are already part of the model.

This does not mean the additional passes are free. Each pass requires computation, intermediate activations and potentially additional attention-cache storage. It means that computational depth no longer needs to increase proportionally with the number of unique parameters.

### Understanding the recurrence mathematically

Let P represent the prelude, R the recurrent transformer core, and C the coda.

h₀ = P(x)

h₁ = R(h₀; θᵣ)

h₂ = R(h₁; θᵣ)

h₃ = R(h₂; θᵣ)

y = C(h₃)

The input to each recurrent pass changes, while its learned parameters remain the same. Repeatedly applying a nonlinear function is not equivalent to executing the same calculation on the same input. Each iteration receives an updated hidden representation and can potentially refine, combine or transform information produced by the previous iteration.

Whether those transformations correspond to meaningful reasoning is an empirical question, not something guaranteed by the architecture.

## 3. The difference between more parameters and more computation

Consider an educational transformer with six prelude layers, twelve recurrent layers and six coda layers. Assume each layer has approximately the same number of parameters.

For r recurrent passes:

Effective layer applications = 12 + 12r.

Physical layers = 24.

With one, three, five and ten passes, the effective depth is 24, 48, 72 and 132 respectively, while unique layer weights remain unchanged.

[Insert Figure 3: depth.png]

*Figure 3. Analytical layer-count illustration. Actual FLOPs and latency need not scale uniformly; arbitrary loop counts are not established as supported settings for the production checkpoint.*

A conventional transformer with 48 equally sized unique layers would require approximately twice as many layer parameters as a 24-layer transformer. A recurrent transformer can execute 48 layer applications while retaining the original 24 physical layers.

This is particularly attractive for consumer hardware. A laptop’s memory capacity is fixed. It cannot suddenly accommodate twice as many model weights because a question becomes difficult. But it can potentially spend additional time executing the weights it already holds.

Instead of asking how large a model we can fit into memory, we can also ask how much useful computation we can extract from that model.

## 4. Looped transformers are not an entirely new idea

Although Inkling 2 Mini makes recurrent depth particularly relevant to local inference, researchers have been investigating related architectures for years.

**2016: Adaptive Computation Time.** [Graves (2016)](https://arxiv.org/abs/1603.08983) investigates how recurrent networks can learn to allocate different amounts of computation to different inputs.

**2018: Universal Transformers.** [Dehghani et al. (2018)](https://arxiv.org/abs/1807.03819) introduce recurrent, parameter-shared transformations operating over sequence representations, bringing recurrence into transformer depth. They investigate whether repeated self-attentive transformations can improve algorithmic generalization.

**2019: Deep Equilibrium Models.** [Bai, Kolter and Koltun (2019)](https://arxiv.org/abs/1909.01377) investigate implicit networks that seek equilibrium representations rather than relying on a fixed stack of independently parameterized layers.

**2025: Huginn.** [Geiping et al. (2025)](https://arxiv.org/abs/2502.05171) investigate scaling inference computation through a recurrent core inside a pretrained language model, without requiring an equivalent increase in generated reasoning tokens.

**2026: Recurrent-depth mixture of experts.** [Huang et al. (2026)](https://arxiv.org/abs/2610.23036), in Towards Looped Models Done Right, investigate recurrent topology, input injection, state organization and mixture-of-experts designs under controlled computational budgets. In one reported comparison, their recurrent MoE uses 75% fewer resident parameters than a feedforward baseline under matched training and inference FLOP budgets, retaining competitive results across several evaluations.

**November 2026: Inkling 2 Mini.** Thinking Machines Lab releases an open-weight multimodal model combining recurrent depth, sparse expert activation and low-precision weights for local deployment.

The research direction is increasingly clear: parameter count and computational depth should be treated as separate architectural design variables.

## 5. What happens inside the recurrent core?

The simplified architecture described earlier is useful for understanding the idea, but Inkling’s actual implementation is more sophisticated than repeatedly calling the same transformer block.

A naive recurrent architecture would simply implement:

```python
hidden = prelude(x)

for _ in range(3):
    hidden = recurrent_core(hidden)

output = coda(hidden)
```

The problem is that repeatedly transforming a representation can cause information from the original input to become degraded or difficult to recover.

Thinking Machines addresses this by supplying the output of the prelude to successive recurrent passes. Later passes additionally receive a message derived from the preceding pass. Learned gates control how this information is combined.

During training, the contribution of the recurrent message is initially suppressed and subsequently learned, helping introduce recurrence into the model ([Thinking Machines Lab, 2026](https://thinkingmachines.ai/news/inkling-2-mini/)).

A conceptual representation is:

hₜ₊₁ = Rθ(hₜ, p, mₜ)

Here, p is the representation computed by the prelude, hₜ is the current hidden representation, mₜ represents information from the previous recurrent pass, and Rθ is the shared recurrent transformation.

This is a mathematical abstraction of the publicly documented information flow, not the exact training implementation. The architecture resembles a computational process that can repeatedly revisit its input while carrying forward intermediate results. That is considerably different from simply stacking more independently parameterized transformer layers.

## 6. Why mixture of experts makes this architecture practical

Recurrent depth alone does not make a language model efficient. If we repeatedly execute an enormous dense network, we still have to perform its expensive matrix operations during every pass.

Inkling combines recurrence with a mixture-of-experts architecture. Instead of executing the entire feedforward network for every token, an MoE layer contains several expert networks and a routing mechanism that selects a subset of them.

Inkling 2 Mini contains 128 experts per MoE layer and routes tokens to eight experts ([Thinking Machines Lab, 2026](https://thinkingmachines.ai/news/inkling-2-mini/)).

[Insert Figure 4: experts.png]

*Figure 4. Conceptual sparse expert routing. Highlighted experts illustrate one possible routing decision, not measured assignments.*

For a token representation x, the router computes expert scores:

r = softmax(Wᵣx)

S = TopK(r, k)

y = Σᵢ∈S r̃ᵢ Eᵢ(x)

Eᵢ denotes an expert network and r̃ᵢ represents its normalized routing weight. Only selected experts need to execute for a particular token. Different tokens can activate different experts. In a recurrent model, a token’s hidden representation changes between passes, so its routing decisions can potentially change as well.

### Total parameters versus active parameters

Inkling has three relevant parameter measurements:

- Total stored parameters: 26.4B.
- Nominal active parameters: 3.4B.
- Effective active parameter applications, including recurrence: 5.4B.

[Insert Figure 5: parameters.png]

*Figure 5. Reported stored and effective active parameter counts. Repeated parameter applications are counted in the effective value.*

The final figure is particularly important for estimating decoding work. Using only the 3.4B number would ignore the repeated execution of the recurrent core. However, even the effective 5.4B count is not a complete FLOP or memory-traffic measurement. Attention, routing, embeddings, quantization kernels and runtime overhead also contribute.

## 7. How Thinking Machines trained Inkling

One of the most interesting aspects of the release is that the final 26.4B model was not simply trained from scratch with its final expert configuration.

The training process involved a larger expert pool followed by compression and recovery.

[Insert Figure 6: training.png]

*Figure 6. Simplified training workflow from the public technical announcement, not a complete reproduction of the internal training pipeline.*

According to [Thinking Machines Lab (2026)](https://thinkingmachines.ai/news/inkling-2-mini/), two-thirds of the original experts were removed during midtraining: 384 experts became 128. A subsequent trillion-token distillation phase helped the remaining experts recover capability before instruction tuning and reinforcement learning. The reported reinforcement learning stage involved 2.2 trillion tokens and 2,000 steps.

This is an important lesson for researchers interested in small language models. The architecture we deploy does not necessarily need to be the architecture with which we begin training. A larger training-time model can potentially discover useful representations and then transfer them into a smaller deployment-time model.

This connects Inkling to network pruning, knowledge distillation, sparse conditional computation, low-precision training and parameter-efficient recurrent architectures. The innovation lies partly in combining these techniques within one production-oriented model.

## 8. Native 4-bit training is different from quantizing a model afterward

Many people running local LLMs download a model that has been quantized after training. The usual pipeline is higher-precision training, post-training, conversion to four-bit weights, then deployment. This is known as post-training quantization.

Inkling takes a different approach. Its expert computations use native NVFP4 precision throughout training, with the final four layers operating at MXFP8 precision, according to the release documentation.

NVFP4 and MXFP8 are low-precision floating-point formats rather than ordinary four-bit and eight-bit integer representations. Training under the intended numerical format allows the model to adapt to the effects of low-precision computation during optimization.

That avoids one particular problem: introducing a new weight-format conversion after training. It does not establish that low-precision computation has no accuracy cost compared with an otherwise identical high-precision model.

### Understanding the memory requirement

Ideal raw weight storage = parameter count × bits per parameter / 8.

For 26.4 billion parameters, idealized storage is 105.6 GB at 32 bits, 52.8 GB at 16 bits, 26.4 GB at 8 bits, and 13.2 GB at 4 bits.

[Insert Figure 7: precision.png]

*Figure 7. Ideal raw weight-storage calculations, excluding metadata and mixed-precision tensors.*

Actual checkpoint sizes differ because they also include scaling metadata, tensors stored at higher precision, alignment requirements and other model components.

As of November 28, 2026, Ollama lists [Inkling 2 Mini](https://ollama.com/library/inkling-2-mini) variants at approximately 19–20 GB, while the standard [Qwen3.8-27B](https://ollama.com/library/qwen3.8) download is approximately 18 GB.

This explains why Inkling cannot simply be treated as a 5.4B model when deciding whether it fits into memory. The inactive experts still exist and need to be available to the routing system.

**Active parameter count determines part of the computation requirement. Total parameter count determines part of the memory requirement.**

## 9. Why local LLM inference is often limited by memory bandwidth

When running a language model locally, one of the largest bottlenecks during single-user autoregressive decoding is moving model weights from memory into the processing units. The GPU cannot multiply matrices until it has accessed their weights.

Estimated decoding throughput = sustained memory bandwidth / estimated weight traffic per token.

For illustrative purposes, assume both models use an idealized average of four bits per active weight. Qwen’s 27B dense parameter count corresponds to approximately 13.5 GB of idealized weight traffic per token. Inkling’s 5.4B effective parameter applications correspond to approximately 2.7 GB.

That produces an idealized fivefold difference in weight traffic. But this is not a measured fivefold speed improvement.

For example, at 250 GB/s theoretical bandwidth and 60% utilization, the simplified estimates are 55.6 tokens/s for Inkling and 11.1 tokens/s for Qwen. These calculations assume identical utilization and ideal weight traffic, rather than measured hardware performance.

[Insert Figure 8: bandwidth.png]

*Figure 8. Idealized weight-traffic model, not a benchmark. Mixed precision, cache behavior, attention and runtime efficiency are excluded.*

Sparse expert computation can introduce overhead through irregular memory access and smaller matrix multiplications. A dense architecture may benefit from highly optimized kernels even while performing more nominal work.

This is why the model with fewer active parameters is not guaranteed to run faster on every device.

## 10. The benchmarks reveal something more interesting than tokens per second

The published comparisons between Inkling and Qwen illustrate why local inference should be evaluated at the task level.

### Coding capability

The reported scores for Inkling and Qwen, respectively, are 48.2% and 42.2% on DeepSWE v1.1; 77.9% and 73.8% on SWE-bench Pro v2 Hard; 73.7% and 70% on SWE-bench Multilingual; and 71.6% and 73% on Terminal Bench 2.1.

[Insert Figure 9: coding.png]

*Figure 9. Reported benchmark scores from the model comparison. Harnesses and settings may vary; these are not independent controlled measurements.*

These reported results indicate competitive coding and agentic performance. See the [Inkling announcement](https://thinkingmachines.ai/news/inkling-2-mini/) and [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the original evaluation conditions.

### What happens on actual Macs?

Independent measurements are considerably more nuanced:

- M5 Max, llama.cpp decoding: Inkling 41.6 tokens/s, Qwen 26.9 tokens/s ([GeekByte, November 26, 2026](https://geekbyte.com/post/inkling-2-mini-vs-qwen3-8-27b-coding-speed-m5-max/)).
- M3 Ultra, Ollama coding prompts: Inkling 52.7 tokens/s, Qwen 113.3 tokens/s ([NerdLevelTech, November 25, 2026](https://mcp.nerdleveltech.com/ar/inkling-2-mini-mac-studio-local-tests)).
- 48GB MacBook Pro, complete code audit: Inkling 17 minutes, Qwen 13 minutes ([Stream, 2026](https://getstream.io/blog/local-video-audit-skill/)).

[Insert Figure 10: mac_speed.png]

*Figure 10. Two independent decoding-throughput comparisons on different devices and runtimes. Compare models within each reported test, not across different test configurations.*

Notice that even the direction of the speed advantage changes across configurations. Inkling is faster in one environment. Qwen is substantially faster in another.

This is not necessarily a contradiction. It demonstrates that inference performance is a property of the entire deployment stack, not just the neural network architecture.

## 11. A faster model can take longer to finish the same task

This is perhaps the most useful engineering lesson from the entire comparison.

Consider two hypothetical language models. Model A generates 20 tokens per second but needs 8,000 tokens to solve a problem. Model B generates 12 tokens per second but needs only 3,000 tokens.

Model A requires 400 seconds of decoding. Model B requires 250 seconds. The model with lower decoding throughput finishes the task sooner.

Total task latency = load time + prefill time + decode time + tool time + retry time.

Decode time = generated token count / decoding throughput.

[Insert Figure 11: task_time.png]

*Figure 11. Hypothetical example illustrating why faster decoding need not mean faster completion.*

The number of generated tokens matters as much as the speed at which they are produced.

[Stream’s independent code-audit experiment](https://getstream.io/blog/local-video-audit-skill/) demonstrates a related trade-off. Across 264 individual checks, Inkling matched a larger reference model’s verdict on 224 occasions, compared with 218 for Qwen. However, the Qwen run finished approximately four minutes sooner. The benchmark measured agreement with Claude Opus 5.5, not independently established correctness for every check.

For an AI engineer, this suggests a better evaluation metric:

Useful throughput = successfully completed tasks / elapsed time.

A model generating more tokens is not necessarily doing more useful work.

## 12. Can recurrent depth replace chain-of-thought tokens?

One of the more scientifically interesting findings from Thinking Machines concerns the relationship between latent computation and explicit reasoning tokens.

The researchers evaluated a separate variable-loop checkpoint, rather than merely comparing two unrelated production models. They investigated what happens when the model receives additional recurrent passes.

For a subset of GPQA Diamond questions answered correctly at every tested loop count, the reported mean reasoning length was 3,703 tokens with one pass, 1,731 with three passes and 1,661 with five passes.

[Insert Figure 12: reasoning.png]

*Figure 12. Reported mean reasoning-token counts for the selected GPQA Diamond subset and variable-loop research checkpoint.*

Increasing the recurrent depth from one to five passes reduced the mean reasoning trace by approximately 55%. This demonstrates a possible trade-off between computation performed internally in hidden representations and computation performed through autoregressively generated intermediate tokens.

These results concern the selected questions and variable-loop research model, not a universal reduction across all workloads ([Thinking Machines Lab, 2026](https://thinkingmachines.ai/news/inkling-2-mini/)).

In a conventional reasoning model, generating intermediate tokens provides additional sequential computation. In a looped model, some of that computation can occur before the next token is emitted.

Approximate total computation = number of tokens × computation per token.

Looping increases computation per token. If it simultaneously reduces the number of tokens needed to solve the task, the overall effect can potentially be positive or negative. That depends on the task and how effectively the model uses its additional latent computation.

## 13. Why repeating layers does not automatically create more factual knowledge

Suppose a language model is asked to solve a mathematical puzzle. Additional computation may allow it to examine more relationships, reconsider intermediate results, or refine its answer.

Now suppose the same model is asked to recall a rare fact that it did not learn during training. Running its layers repeatedly does not introduce new external information.

This creates a useful distinction between knowledge capacity and computational capability. The variable-loop experiments reported by Thinking Machines found stronger benefits on multi-hop reasoning and long-context tasks than on several closed-book factual evaluations.

This does not mean computation can never help factual retrieval. Additional processing may help access knowledge already represented in the weights or infer facts from related information. But recurrent execution does not automatically expand the amount of information available to the model.

[Insert Figure 13: grounding.png]

*Figure 13. Combining external knowledge retrieval with local recurrent computation.*

Instead of trying to store every relevant fact inside a larger language model, a system can retrieve external information and use the model’s computational capabilities to work with it. This is particularly relevant to coding agents, which can retrieve documentation, inspect repositories and execute tests rather than rely entirely on memorized knowledge.

## 14. The context-window problem: why one million tokens does not mean you can use them on a laptop

Inkling 2 Mini supports a context window of approximately one million tokens. Qwen3.8-27B has a native context window of 262,144 tokens. However, a model’s supported context length is not the same as the context length that can be efficiently deployed on a particular computer.

Total memory = weights + KV cache + activations + runtime overhead.

For conventional full attention, the simplified cache requirement in bytes is:

Mₖᵥ = 2 × L × T × Hₖᵥ × dₕ × b

L is the number of cached attention-layer applications, T is context length, Hₖᵥ is the number of key-value heads, dₕ is head dimension, and b is bytes per stored element. The factor of two represents keys and values.

For a looped transformer, shared weights do not automatically imply shared KV caches. Different recurrent passes can produce different keys and values from different hidden states. Consequently, cache behavior depends on the architecture and how the runtime implements it.

### Sliding-window attention

Inkling uses a combination of local and global attention layers. Local attention allows a token to attend directly to a limited region of preceding tokens. Global attention provides access across the broader context.

For an idealized hybrid attention architecture:

Mₖᵥ = 2 × Hₖᵥ × dₕ × b × [LᵍT + Lˡ min(T, w)]

Lᵍ is the count of global attention-layer applications, Lˡ is the local count, and w is the local attention window.

[Insert Figure 14: attention.png]

*Figure 14. Illustrative full causal and sliding-window attention masks. Hybrid models combine local and global attention.*

This can reduce cache growth if the implementation retains only the necessary local history. However, we cannot responsibly compare the exact memory footprint of Inkling and Qwen using their context lengths alone. Qwen also uses efficient recurrent attention mechanisms, and both models have architecture-specific cache behavior.

The safest approach is to measure memory consumption under the actual runtime and quantized checkpoint you intend to use.

## 15. Implementing a miniature looped transformer in PyTorch

We do not need billions of parameters to understand how recurrent depth works. We can construct a small transformer consisting of token and positional embeddings, a prelude transformer block, a recurrent transformer block, a coda transformer block and a language-model prediction head.

The important architectural property is that the same recurrent block is reused.

```python
import torch
import torch.nn as nn


class LoopedTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=128,
                 n_heads=4, max_seq_len=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        def make_block():
            return nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )

        self.prelude = make_block()
        self.recurrent_core = make_block()
        self.coda = make_block()
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, loops=3):
        if loops < 1:
            raise ValueError("loops must be positive")
        _, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool,
                       device=tokens.device),
            diagonal=1
        )
        x = self.prelude(x, src_mask=mask)
        for _ in range(loops):
            x = self.recurrent_core(x, src_mask=mask)
        x = self.coda(x, src_mask=mask)
        return self.lm_head(self.norm(x))
```

This implementation demonstrates recurrent parameter sharing using standard PyTorch components. Although TransformerEncoderLayer is the primitive being used, the explicit causal mask prevents attention to future positions.

It is not a reproduction of Inkling’s full architecture. We intentionally omit mixture of experts, native low-precision training, specialized cross-pass message gates, vision encoders and production KV caching.

### Verify that increasing computation does not increase parameter count

```python
model = LoopedTransformer()
tokens = torch.randint(0, 256, (4, 64))
parameter_count = sum(p.numel() for p in model.parameters())

for loops in [1, 2, 3, 5, 10]:
    output = model(tokens, loops=loops)
    print(f"Loops: {loops}, Parameters: {parameter_count:,}, "
          f"Output: {tuple(output.shape)}")
```

Every loop count uses the same model parameters. The amount of executed computation changes because we repeatedly call the recurrent block. This illustrates the central architectural property behind recurrent-depth models.

## 16. Adding a miniature mixture-of-experts layer

The next step is to replace a conventional feedforward network with a sparse expert layer.

```python
import torch.nn.functional as F


class SparseMoE(nn.Module):
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError("Invalid top_k")
        self.top_k = top_k
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            ) for _ in range(n_experts)
        ])

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        router_logits = self.router(x)
        scores, indices = torch.topk(router_logits, self.top_k, dim=-1)
        weights = F.softmax(scores, dim=-1)
        output = torch.zeros_like(x)

        for expert_id, expert in enumerate(self.experts):
            token_ids, slot_ids = torch.where(indices == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_output = expert(x[token_ids])
            expert_weight = weights[token_ids, slot_ids].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_output * expert_weight)

        return output.reshape(original_shape)
```

The router selects two of eight experts for each token representation. Unlike a demonstration that evaluates every expert and then masks unnecessary outputs, this implementation dispatches only the selected token representations to each expert.

A production implementation would require more efficient dispatch, grouped matrix multiplication, balancing objectives, fused kernels and other hardware-specific optimizations.

The educational value is that we can combine sparse activation across available experts and repeated execution across recurrent depth. These are complementary rather than competing mechanisms. The two code examples are standalone building blocks, not an integrated Inkling implementation.

## 17. How should we train a looped transformer?

A major research question is whether a recurrent model trained with a fixed loop count will continue improving when more iterations are introduced at inference time. We should not assume that it will.

A model trained exclusively with three recurrent passes may not behave reliably when executed with ten passes. Repeated nonlinear transformations can produce instability, representation collapse or diminishing improvements.

One possible educational training strategy is to sample the recurrent depth during training.

```python
import random

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


def train_step(batch):
    model.train()
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    loops = random.choice([1, 2, 3, 4, 5])
    logits = model(inputs, loops=loops)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {"loss": loss.item(), "loops": loops}
```

This is an illustrative variable-depth training recipe, not Thinking Machines’ original training procedure.

The appropriate experimental comparison would include independently parameterized dense baselines, shared recurrent blocks trained at fixed depth, and shared recurrent blocks trained at variable depth. It would also be necessary to control for total training FLOPs. Otherwise, a model could appear superior simply because it received more computation during training.

### A good task for researching recurrent depth

A particularly useful synthetic benchmark is pointer chasing. Suppose we provide a model with the following relationships:

A → D, B → E, C → A, D → F, E → C, F → B.

Then ask: Starting at A, follow three links. Which node do you reach?

The answer is B.

Increasing the number of required hops creates an explicit sequential reasoning problem. The experiment should evaluate whether additional recurrent passes allow a model to solve longer chains, especially chains exceeding those encountered during training. This is conceptually related to multi-hop reasoning investigations in Thinking Machines’ release and the broader recurrent-depth literature.

## 18. Running Inkling and Qwen locally on a Mac

Both models are available through Ollama. Install a current compatible version from [ollama.com](https://ollama.com/download).

Check the installed version:

```bash
ollama --version
```

Download and run Inkling:

```bash
ollama run inkling-2-mini
```

Download and run Qwen:

```bash
ollama run qwen3.8:27b
```

Inspect installed models and models currently loaded into memory:

```bash
ollama list
ollama ps
```

[Ollama’s Inkling page](https://ollama.com/library/inkling-2-mini) documents support for Macs with at least 32GB of memory and includes integrations for Claude Code and other agent applications.

### Practical Mac memory recommendations

- **16GB:** Prefer substantially smaller models.
- **24GB:** Consider smaller or more aggressively quantized variants.
- **32GB:** Reasonable starting point for standard Inkling, with controlled context.
- **48GB:** More comfortable for local coding and longer contexts.
- **64GB and above:** Greater flexibility for cache allocation and higher-precision variants.
- **96GB and above:** Useful for larger contexts, experiments and multiple workloads.

These are practical estimates rather than guaranteed runtime requirements. Start with a relatively small context allocation, such as 8K or 16K tokens, and expand it after measuring memory pressure and prompt-processing time.

Do not allocate a one-million-token context merely because the model advertises support for it.

## 19. Benchmark both models yourself

Instead of relying on somebody else’s hardware measurements, we can build a simple local benchmarking script.

The Ollama API provides generation counts and timing information that can be used to measure prompt-processing speed, decoding speed and overall latency. The following implementation uses Python’s standard library.

```python
import json
import time
import urllib.request


def benchmark_model(model, prompt, num_predict=2048, num_ctx=8192):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx
        }
    }
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    wall_time = time.perf_counter() - start
    output_tokens = result.get("eval_count", 0)
    decode_seconds = result.get("eval_duration", 0) / 1e9
    prompt_tokens = result.get("prompt_eval_count", 0)
    prompt_seconds = result.get("prompt_eval_duration", 0) / 1e9
    return {
        "model": model,
        "wall_seconds": wall_time,
        "output_tokens": output_tokens,
        "decode_tokens_per_second": (
            output_tokens / decode_seconds if decode_seconds else None
        ),
        "prompt_tokens_per_second": (
            prompt_tokens / prompt_seconds if prompt_seconds else None
        ),
        "load_seconds": result.get("load_duration", 0) / 1e9,
        "done_reason": result.get("done_reason"),
        "response": result.get("response"),
        "thinking": result.get("thinking")
    }
```

We can then evaluate both models on the same prompt.

```python
prompt = """
Implement an LRU cache in Python.

Requirements:
1. O(1) get and put operations.
2. Use a dictionary and a doubly linked list.
3. Handle a capacity of zero.
4. Include unit tests.
"""

models = ["inkling-2-mini", "qwen3.8:27b"]
results = []

for model_name in models:
    result = benchmark_model(model_name, prompt)
    results.append(result)
    print(json.dumps(result, indent=2))

with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

For a scientifically useful comparison, repeat each workload several times and record the model checkpoint and quantization variant, Ollama version and hardware configuration, reasoning settings, prompt-processing rate, time to first token using streaming, generated token count, total wall-clock latency, peak memory usage, and whether the generated solution passes independent tests.

The sample script measures nonstreaming latency and aggregate generation statistics. Measuring time to first token or peak memory requires additional instrumentation. The most important evaluation criterion is whether the generated code actually works.

## 20. Which model should an AI engineer choose?

The answer depends on the application rather than a single aggregate benchmark.

**Inkling 2 Mini:** Consider the recurrent-depth MoE for grounded coding tasks, multi-hop reasoning, research agents, document analysis and experiments with recurrent computation.

**Qwen3.8-27B:** Consider the dense multimodal architecture for mature local deployment workflows, established integrations, general conversation and workloads where your measured end-to-end latency is lower.

For an AI coding agent, I would evaluate both models on actual repository tasks: fixing a bug across multiple files, writing and passing unit tests, understanding unfamiliar APIs using supplied documentation, reviewing a pull request, performing a controlled refactoring, and following a multi-step tool-use procedure.

I would not choose a model solely because it produced text faster on a short benchmark.

For agentic coding, filesystem permissions also matter. Keep the agent inside an isolated workspace or Git branch, review proposed changes, and restrict destructive operations. A capable local model should not automatically receive unrestricted control over your machine.

## 21. What looped transformers mean for interpretability research

There is another important scientific consequence of recurrent depth. In a conventional transformer, a researcher may inspect the activation associated with a particular physical layer and sequence position.

In a looped transformer, the same physical layer can execute multiple times on different intermediate representations. A physical layer index is no longer sufficient to uniquely identify a computational event. We also need the recurrent pass index.

Conceptually, layer 10 on pass one, layer 10 on pass two and layer 10 on pass three are distinct computational events. The weights may be identical, but the activations need not be.

[Goodfire’s preliminary interpretability investigation of Inkling](https://www.goodfire.com/research/interpreting-inkling-2-mini/) examines communication between layers and recurrent passes, including the role of cross-loop messages and methods for decoding intermediate representations.

[Insert Figure 15: interpretability.png]

*Figure 15. Conceptual cross-pass information flow. Physical layer identity is distinct from position in the unrolled computation.*

This raises several interesting research questions. Do particular layers develop different computational roles during successive passes? Can a feature written during one iteration influence a physically earlier layer during the next iteration? Are sparse autoencoders trained on one loop’s activations transferable to another loop?

These are important extensions of existing mechanistic interpretability techniques. They also connect directly to linear factor models and sparse representations: the same decoder weights can be associated with different internal representations depending on the computation that produced them.

## 22. Where this research could go next

Inkling uses a fixed three-pass recurrent structure in its released production model. An interesting future direction is adaptive computational depth.

Instead of allocating the same amount of computation to every token, a model could learn when additional passes are necessary.

Imagine two requests: “What is 2 + 2?” and “Find the concurrency bug in this distributed transaction-processing system.” These tasks clearly do not require the same amount of computation.

A future architecture could potentially allocate additional recurrent passes when a learned halting mechanism predicts that more processing would be useful.

One possible halting probability is:

pₜ = σ(wᵀhₜ + b)

The model could continue iterating until a learned stopping criterion is satisfied. However, training this mechanism is difficult. It needs to balance computational cost against accuracy and avoid halting prematurely on difficult cases or repeatedly executing unnecessary computation on easy ones.

This connects modern recurrent-depth models with earlier adaptive computation research, selective prediction and learned decision systems.

For engineers interested in agent architecture, there is an even broader possibility. A model could learn to decide between performing another internal recurrent pass, generating an explicit reasoning token, retrieving external evidence, executing a tool, or delegating the task to another model.

Each of those actions represents a different way of allocating computational resources.

## Conclusion

The most interesting thing about Inkling 2 Mini is not simply that another capable language model can run on a laptop.

It is that its architecture challenges the assumption that increasing model capability necessarily requires increasing the number of unique parameters.

Mixture of experts allows us to select which stored weights participate in a computation. Recurrent depth allows us to decide how often those weights are executed. Low-precision training allows us to reduce the storage and numerical cost of those weights.

Together, these techniques create new possibilities for designing capable models under constrained hardware budgets. But none of them eliminates the need for rigorous evaluation.

A model that produces tokens faster can still take longer to complete a task. A model with fewer active parameters can still require substantial memory. A model that performs additional internal computation can still lack the factual information needed to answer a question correctly.

The future of efficient AI may not simply be about fitting larger models into our computers.

It may be about finding better ways to use the models that already fit.

**Research and implementation limitations.** The architectural descriptions and benchmark numbers above are based on the cited public sources and independent reports, not original production-model measurements. The PyTorch examples are educational, not a reproduction of Inkling. No training or Mac hardware benchmarking was performed for this article. The calculations and conceptual illustrations should not be treated as measured performance. A useful next experiment would compare parameter-matched and compute-matched dense and recurrent transformers on multi-hop tasks, with an executable local inference benchmark.

## Research references and further reading

### Foundational architecture research

1. Graves, A. (2016). [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983).
2. Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
3. Shazeer, N., et al. (2017). [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538).
4. Dehghani, M., et al. (2018). [Universal Transformers](https://arxiv.org/abs/1807.03819).
5. Bai, S., Kolter, J. Z., and Koltun, V. (2019). [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377).
6. Beltagy, I., Peters, M. E., and Cohan, A. (2020). [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
7. Fedus, W., Zoph, B., and Shazeer, N. (2022). [Switch Transformers](https://arxiv.org/abs/2101.03961).
8. Dao, T., et al. (2022). [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).

### Recurrent depth and modern architectures

9. Geiping, J., et al. (2025). [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171).
10. Huang, B., et al. (2026). [Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design](https://arxiv.org/abs/2610.23036).
11. Thinking Machines Lab. (2026). [Introducing Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/).
12. Thinking Machines Lab. (2026). [Inkling 2 Mini Model Card](https://thinkingmachines.ai/model-card/inkling-2-mini/).
13. Thinking Machines Lab. (2026). [Inkling 2 Mini Open Weights](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
14. Qwen Team. (2026). [Qwen3.8-27B Model Card](https://huggingface.co/Qwen/Qwen3.8-27B).
15. Goodfire. (2026). [Interpreting Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini/).

### Quantization, pruning and distillation

16. Hinton, G., Vinyals, O., and Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).
17. Han, S., Mao, H., and Dally, W. J. (2016). [Deep Compression](https://arxiv.org/abs/1510.00149).
18. Frantar, E., et al. (2023). [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323).
19. Lin, J., et al. (2024). [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978).

### Independent benchmarks and deployment

20. Stream. (2026). [Run a Code Audit Skill Locally: Qwen3.8 vs Inkling 2 Mini](https://getstream.io/blog/local-video-audit-skill/).
21. ModelFit. (2026). [The Qwen3.8 Successor? Inkling 2 Mini Tested on Mac](https://modelfit.io/blog/inkling-2-mini-mac-benchmark/).
22. GeekByte. (2026). [Inkling 2 Mini vs Qwen3.8 27B: Coding and Speed on M5 Max](https://geekbyte.com/post/inkling-2-mini-vs-qwen3-8-27b-coding-speed-m5-max/).
23. NerdLevelTech. (2026). [Inkling 2 Mini on Mac Studio: Real Local Tests](https://mcp.nerdleveltech.com/ar/inkling-2-mini-mac-studio-local-tests).
24. Ollama. [Inkling 2 Mini](https://ollama.com/library/inkling-2-mini).
25. Ollama. [Qwen3.8](https://ollama.com/library/qwen3.8).
