# Google Searched 25,000 Transformer Programs. Two Small Changes Made Language Models Much Cheaper to Train.

### Inside Primer EZ: squared ReLU, convolutional attention projections, an independent PyTorch implementation, and a surprising failure when the sequence pattern changes

*An engineering tutorial for machine learning researchers and developers. Based on the NeurIPS 2021 paper "Primer: Searching for Efficient Transformers for Language Modeling" by David R. So and colleagues.*

---

There is something strange about the way we optimize neural networks.

We will spend weeks designing a distributed training stack, choosing a new optimizer, implementing kernel fusion, profiling memory bandwidth, and deciding whether to double the GPU count. Yet the actual function executed millions of times inside a Transformer can remain almost untouched.

What if the most valuable optimization is neither a faster GPU kernel nor a bigger model?

What if it is a different activation function, followed by three tiny convolutions?

This was the question behind Google's Primer research. The researchers did not start with a grand theory of what attention should become. Instead, they let an evolutionary search modify the lower-level operations that form a Transformer block. The search explored roughly 25,000 candidate programs. The researchers then inspected the successful designs and distilled their most transferable ideas into a much smaller recipe called **Primer EZ**. [1][2]

The two ingredients were unusually modest:

1. Replace the feed-forward network's ReLU with `relu(x) ** 2`.
2. Apply a three-position, causal, depth-wise convolution after each query, key, and value projection, before calculating attention.

In large-scale experiments reported by the authors, the full Primer architecture reached the quality of a 500-million-parameter autoregressive T5 baseline with roughly **4.2 times less training compute**. A 1.9-billion-parameter comparison reached similar one-shot performance using roughly **one-third of the baseline's pretraining compute**. [1][2]

But the phrase *more efficient* needs careful handling. These modifications can actually make an individual training step **slower**. They improve how quickly the model learns useful structure, allowing it to reach a desired loss after fewer steps. That distinction will matter throughout this article.

And when we implemented the modifications ourselves, we found a useful complication: the model that learned our training distribution best was also the one that failed most severely under one structural distribution shift. The lesson is not that Primer is fragile. It is that architecture optimization, generalization, and compute efficiency have to be measured separately.

The rest of this tutorial shows how.

---

## Table of contents

1. Why the objective is time to quality, not time per step
2. What Google's evolutionary search actually searched
3. Primer versus Primer EZ
4. Reconstructing the original Transformer block
5. Squared ReLU from first principles
6. Why a squared activation changes optimization
7. What Multi-DConv-Head Attention changes
8. Causal convolution and the easiest leakage bug to introduce
9. Depth-wise groups and a subtle head-sharing choice
10. Building the modules in PyTorch
11. Implementing causal attention without shortcuts
12. Assembling a decoder-only language model
13. Constructing a controlled language-modeling dataset
14. Experimental protocol and fair baselines
15. The first ablation: feed-forward activations
16. The second ablation: local convolution
17. Full Primer EZ versus vanilla attention
18. What happens when the sequence period changes
19. Isolating Q, K, and V projections
20. Per-head convolution versus shared convolution
21. Equal steps versus approximately equal wall-clock time
22. Understanding perplexity and accuracy
23. What the results do not establish
24. Why large-scale paper results can differ from small-model results
25. Testing that future tokens remain invisible
26. A reproducible repository you can run
27. How to adapt the experiment to real text
28. GPU profiling and optimized attention kernels
29. Better experiments for serious replication
30. What Primer teaches us about architecture search
31. References and further reading

---

## 1. Why the objective is time to quality, not time per step

Imagine two language models trained on the same dataset.

Model A completes an optimization step in 10 milliseconds. Model B takes 15 milliseconds because it performs a few additional operations.

If A needs 10,000 steps to reach your target perplexity, training takes 100 seconds. If B needs only 3,000 steps, training takes 45 seconds.

Model B is the slower model per step. Yet it is more than twice as efficient at reaching the specified quality target.

This is exactly the sort of tradeoff that architecture-search benchmarks often obscure. A runtime microbenchmark tells you how quickly the machine can execute a computation. It does not tell you how much useful learning each execution produces.

For research, keep four measurements separate:

- **Step latency:** seconds per optimizer step, at a specified batch size and sequence length.
- **Sample efficiency:** validation loss after a specified number of examples or tokens.
- **Compute to quality:** hardware time or estimated operations needed to reach a predefined loss target.
- **Inference economics:** latency, memory, and throughput after training, at fixed model quality.

Primer's main claim concerns the third category. Google explicitly notes that its strongest modifications increase step time while improving the model's learning efficiency enough to reduce total training cost. [2]

A fifth metric will become central in our own experiment: **performance outside the training distribution**. A model can dominate an in-distribution loss curve without learning the rule we hoped it had learned.

## 2. What Google's evolutionary search actually searched

A common misconception is that Primer's researchers tried a list of existing activation functions and selected the winner. The search was more ambitious.

Instead of treating a Transformer block as an indivisible unit, their system described a stackable decoder block as a program made from lower-level TensorFlow primitives. A candidate architecture was a collection of instructions, each referring to earlier intermediate tensors and using operations such as addition, multiplication, nonlinear transformations, matrix multiplication, and learned projections. [2]

The search process worked approximately like this:

1. Represent a decoder architecture as a sequence of typed program instructions.
2. Compile the instructions into a TensorFlow computation graph.
3. Reject constructions that fail or cannot train.
4. Train surviving candidates under a limited accelerator budget.
5. Measure validation perplexity.
6. Select, mutate, and replace architectures using regularized evolution.
7. Test promising designs at greater compute scales and in additional codebases.

The key insight is that the search could synthesize modifications the researchers had not explicitly included as high-level architectural options.

The published search used a fixed-budget objective associated with a 24 TPUv2-hour LM1B training task. To reduce the extraordinary cost of evaluating candidates, it used early performance hurdles and proxy training budgets. The authors report examining approximately 25,000 individuals. [2]

The search did not emerge from random initialization alone. Most unconstrained primitive programs are not useful neural networks. The researchers therefore seeded the evolutionary population with Transformer-like programs, a technique they call **conceptual initialization**.

That is a subtle form of inductive bias. The search space was relatively open, but the starting point already contained useful architectural knowledge.

This matters when interpreting the result. Primer is neither a randomly invented architecture nor a manually designed attention variant. It is a discovered program whose successful parts were subsequently studied by human researchers.

## 3. Primer and Primer EZ are not the same thing

The original search produced a model called **Primer**, short for PRIMitives searched transformER. Its discovered program contained additional changes involving normalization, representation sharing, bottleneck sizes, and other graph operations.

The researchers then investigated which changes transferred reliably across different training setups. They found that two contributed much of the benefit and were comparatively easy to adopt. The simplified combination is **Primer EZ**. [2]

Why simplify a model that won an architecture search?

Because a complicated winning configuration can contain properties that help only in the exact search environment. A smaller set of robust changes is more useful to practitioners than a long list of tricks whose effects depend on particular hyperparameters.

Our implementation is a **Primer-EZ-inspired decoder**, not an exact replica of the full searched Primer network. It uses ordinary pre-normalized Transformer blocks and replaces only the components identified in the simplified recipe.

This is also how the labml.ai annotated implementation approaches the method. It exposes `SquaredReLU` and a convolutional attention projection as small, comprehensible PyTorch modules. [3][4]

## 4. Reconstructing the original Transformer block

Before modifying anything, we need a stable baseline.

A small causal Transformer typically follows this path:

```text
Token indices
    |
Token embeddings + position embeddings
    |
    v
LayerNorm
    |
Q, K, V linear projections
    |
Causal self-attention
    |
Output projection
    |
Residual addition
    |
LayerNorm
    |
Linear -> ReLU -> Linear
    |
Residual addition
    |
Next layer
    |
Output logits
```

Three different mechanisms are involved.

First, attention transports information between token positions. A representation at position 12 can use earlier information from positions 0 through 12, but not positions 13 onward.

Second, the feed-forward network transforms each token's embedding independently after attention has updated it.

Third, residual connections preserve an information pathway across the block.

Primer EZ does not remove self-attention. It also does not replace the Transformer with a convolutional model. Its convolution changes what the **attention projections** know before they are compared or combined.

That distinction is important: a three-position convolution inside each head has a small local receptive field, while the subsequent self-attention operation can still connect distant tokens.

## 5. Squared ReLU from first principles

Ordinary ReLU is one of the simplest functions in deep learning. It returns zero for negative inputs and leaves positive inputs unchanged.

Squared ReLU makes one additional change: it squares the positive output.

```python
import torch
from torch import nn


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(x)
        return y * y


activation = SquaredReLU()
print(activation(torch.tensor([-2., -1., 0., 1., 2., 3.])))
# tensor([0., 0., 0., 1., 4., 9.])
```

There are no new weights, gates, attention heads, or normalization layers. The transformation introduces no trainable parameters.

Its effect on representation geometry is more meaningful than its line count suggests.

A positive activation of 1 becomes 1. A positive activation of 2 becomes 4. A positive activation of 3 becomes 9.

Instead of increasing linearly, the strongest positive activations become disproportionately influential.

One way to think about this is feature competition. When several hidden features are active, squared ReLU allows a few strong features to contribute more heavily to the feed-forward output. Whether that is useful depends on the data distribution, initialization, optimization, and surrounding network.

It should not be described as simply "a better ReLU." If the method were universally superior, selecting an activation would be trivial. The original paper's comparisons are specific to particular Transformer training regimes and model families.

## 6. Why a squared activation changes optimization

For negative inputs, Squared ReLU produces zero and has zero derivative. For positive inputs, its derivative increases in proportion to the input.

For example:

```python
x = torch.tensor([-2., 0., 3.], requires_grad=True)
loss = SquaredReLU()(x).sum()
loss.backward()
print(x.grad)
# tensor([0., 0., 6.])
```

This has two opposing implications.

The positive implication is that strong activations can receive larger gradient signals, which can change how quickly the network separates useful features.

The negative implication is that sufficiently large activations can create unusually large gradients. A successful training run therefore does not justify dropping normal stabilization practices such as loss monitoring, sensible learning rates, normalization, and gradient clipping.

Squared ReLU also has an interesting relationship to gated feed-forward networks. A ReGLU unit multiplies one projection by the rectified output of another projection. If those projections share the same weights and there are no additional transformations between them, the multiplication becomes similar to squaring the rectified projection.

This is a useful conceptual relationship, not an identity between arbitrary ReGLU and Squared-ReLU networks. Modern SwiGLU and other gated units use independent projections and different activation properties. [5]

In other words, Squared ReLU adds some multiplicative behavior without introducing the extra projection of an ordinary gated FFN.

That is one reason the change was attractive for efficient model design.

## 7. Multi-DConv-Head Attention changes what Q, K, and V contain

Consider the ordinary query projection at a single sequence position.

The model takes that position's hidden state and applies a learned matrix. The resulting query represents information derived from **that position's current representation**.

Some information from earlier tokens may already be present because previous Transformer layers have mixed the sequence. But the query projection itself is position-wise.

Primer EZ inserts a width-three, depth-wise convolution after the projection.

```text
Input residual stream
        |
   linear projection
        |
  split into heads
        |
causal depth-wise Conv1d
        |
       Q heads

The same pattern is used for K and V.
```

At time step `t`, the modified projection can combine projected information from positions `t`, `t-1`, and `t-2`.

What does that buy us?

A query no longer has to depend solely on its position's projected vector. It can carry a short summary of local history. Keys can encode a locally enriched description of the token positions available for matching. Values can carry a similarly enriched payload for attention to retrieve.

The operation is **depth-wise**. Each feature channel has its own tiny convolution kernel. Channels are not mixed by the convolution itself. Channel mixing comes from the linear projections and subsequent attention layers.

The order is also important. This is not an ordinary convolution on the raw embedding followed by attention. The model first learns task-dependent projections and then mixes the resulting channel values across nearby sequence positions.

Google's ablations found this placement more useful than several alternative convolution configurations. [2]

## 8. Causality: the easiest bug in this implementation

Autoregressive language models must not use future tokens to predict an earlier token.

That sounds obvious until you introduce a convolution.

A common `Conv1d` implementation uses symmetric padding so that the output length matches the input length. For a kernel of size three, the output at a particular position may include the next position unless padding and cropping are handled carefully.

In a next-token prediction task, that future position can contain the answer being predicted. A leaking convolution may therefore produce spectacular validation accuracy for the wrong reason.

There are two ways to implement causal width-three convolution:

- Pad only on the left with two zeros, then perform an unpadded convolution.
- Use symmetric padding and crop the extra output on the right, as in the provided labml.ai implementation. [3]

Our lab uses the first version because the causal dependency is particularly easy to inspect.

```python
import torch.nn.functional as F

# Input to Conv1d: [batch, channels, sequence]
left_padded = F.pad(projected, (kernel_size - 1, 0))
causal_output = depthwise_conv(left_padded)
```

For a simple width-three kernel whose coefficients are all one, the input `[1, 2, 3, 4]` must become `[1, 3, 6, 9]`.

If the first result depends on the third input, the implementation is not causal.

A second, stronger test changes all future tokens after a given position and verifies that earlier model outputs are unchanged. We perform that test for the convolution, attention, and complete language model.

## 9. Depth-wise convolution has a head-sharing detail worth noticing

The annotated labml.ai code expects a tensor of shape `[sequence, batch, heads, head_dim]`.

It rearranges that tensor into `[batch * heads, head_dim, sequence]`, then uses a grouped `Conv1d` with `groups=head_dim`.

This produces a separate kernel for each channel **within a head**, but the same set of channel kernels is reused across heads because the head dimension has been folded into the batch dimension.

That is a valid and useful compact implementation. It is not, however, the only interpretation of applying depth-wise convolution to every attention head.

For a four-head model with 12 features per head and a width-three kernel:

- **Shared across heads:** 12 kernels per Q, K, or V branch.
- **Different kernels for every head:** 48 kernels per branch.

Our project supports both versions. The default matches the labml.ai sharing pattern, while an additional `primer_per_head` configuration measures the effect of removing that sharing.

This comparison is worth doing because architectural descriptions sometimes hide parameter-sharing decisions in a tensor reshape. The model's mathematical family has not changed, but its parameter budget and inductive bias have.

## 10. Building causal depth-wise convolution in PyTorch

Here is the heart of our independent implementation, simplified to the shared-kernel variant.

```python
import torch
from torch import nn
from torch.nn import functional as F


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, heads: int, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=kernel_size,
            groups=head_dim,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, sequence, head_dim]
        batch, heads, seq, channels = x.shape

        x = x.permute(0, 1, 3, 2)
        x = x.reshape(batch * heads, channels, seq)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)

        x = x.reshape(batch, heads, channels, seq)
        return x.permute(0, 1, 3, 2).contiguous()
```

The critical line is `groups=head_dim`. That is what makes this convolution depth-wise rather than a fully mixed convolution. PyTorch's `Conv1d` documentation describes exactly this grouped-channel behavior. [6]

For a model with 48 hidden dimensions and four heads, each head has 12 dimensions. A width-three depth-wise convolution contains 36 kernel weights and 12 biases.

Since Primer EZ adds one of these to Q, K, and V, each attention layer adds 144 convolution parameters in the shared variant.

That is tiny compared with the thousands of weights already used in its attention and feed-forward projections.

One implication is worth keeping in mind: if a model changes dramatically when adding 144 learned parameters, the gain is not explained simply by making the network much larger. It is more likely related to the structure of the operation and how it changes information flow.

## 11. Implementing the attention operation

The next step is to insert those convolutions into the ordinary query, key, and value paths.

Our full source code contains shape validation and switches for each projection. Conceptually, the operation looks like this:

```python
# x: [batch, sequence, d_model]
q, k, v = project_and_split_qkv(x)

# Each tensor: [batch, heads, sequence, head_dim]
q = q_conv(q)
k = k_conv(k)
v = v_conv(v)

scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)

# True above diagonal means future attention is forbidden.
future = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)

weights = scores.softmax(dim=-1)
out = weights @ v
```

Notice the order. The convolutions act before the dot products, so the model compares locally enriched queries and keys.

It would be a different architecture if we convolved the already-computed attention outputs instead.

The explicit attention implementation is intentional. Modern PyTorch provides optimized scaled-dot-product attention, which is usually the better choice for production training. But the unfused implementation makes this tutorial auditable: we can inspect each intermediate tensor and test exactly where the causal boundary is enforced. [7]

An important engineering principle follows: first validate the reference computation, then replace it with an optimized kernel and check numerical equivalence.

## 12. Assembling the language model

We now have all required parts.

Our tiny decoder has:

- A 16-symbol token vocabulary.
- Context length 24.
- Hidden dimension 48.
- Four attention heads.
- Two Transformer blocks.
- Feed-forward dimension 144.
- Learned token and positional embeddings.
- Tied token embeddings and output projection weights.
- Layer normalization, residual connections, and AdamW optimization.

Six main architectural options are provided:

```python
ABLATIONS = {
    "baseline_relu":      {"activation": "relu",         "use_dconv": False},
    "squared_relu_only":  {"activation": "squared_relu", "use_dconv": False},
    "dconv_only":         {"activation": "relu",         "use_dconv": True},
    "primer_ez":          {"activation": "squared_relu", "use_dconv": True},
    "gelu_baseline":      {"activation": "gelu",         "use_dconv": False},
    "primer_per_head":    {"activation": "squared_relu", "use_dconv": True},
}
```

The complete project adds three diagnostic variants: Q-only, K-only, and V-only convolution.

The main experiment does **not** reproduce Google's 500-million-parameter results. The purpose is to make the two architectural modifications measurable on hardware available to an individual developer.

The project is CPU-friendly, contains no model downloads, and does not rely on proprietary data.

## 13. Constructing a controlled language-modeling dataset

It is tempting to begin with an enormous natural-language corpus. That sounds more realistic but makes mechanism testing difficult.

When a model improves on a complex dataset, the cause could be token frequency, document boundaries, optimizer settings, regularization, semantic patterns, or chance differences between seeds.

We instead construct a deliberately simple next-token prediction task.

Each sequence begins with a short random motif. After that, the sequence tends to repeat the motif, with a small probability of substituting a random token. A motif of period four might look like:

```text
2, 7, 3, 11, 2, 7, 3, 11, 2, 7, 3, 11, ...
```

The model receives every token except the last and must predict the following token at each position. Its attention remains causal.

In the training and ordinary validation distribution, the period is four. In the distribution-shift evaluation, the period becomes six.

```text
Training pattern:  2, 7, 3, 11, 2, 7, 3, 11, ...
Shifted pattern:   2, 7, 3, 11, 9, 5, 2, 7, 3, 11, 9, 5, ...
```

Each sample has freshly drawn token values. Memorizing specific sequences is not useful. The model has to learn some form of local or positional copying.

The dataset is not intended to simulate natural language. It isolates one question: **how does adding learned local structure before attention affect learning and transfer when the relevant temporal distance changes?**

The implementation also makes the shift inspectable:

```python
# The generator uses a dedicated random seed.
result[:, :period] = torch.randint(vocab_size, (batch, period), generator=gen)

for index in range(period, context + 1):
    override = torch.rand(batch, generator=gen) < noise
    replacement = torch.randint(vocab_size, (batch,), generator=gen)
    result[:, index] = torch.where(
        override, replacement, result[:, index - period]
    )
```

The occasional substitutions prevent the task from being perfectly deterministic. They also create a meaningful lower bound for achievable uncertainty.

## 14. Experimental protocol and fair baselines

The default benchmark uses three independently initialized seeds, each trained for 260 optimizer steps. Every configuration receives the same seed-specific training-data stream and the same seed-specific held-out evaluation samples.

The important constants are:

```text
Model size:        48 hidden dimensions, 2 layers, 4 heads
Sequence length:   24 context positions
Vocabulary size:   16 tokens
Training batch:    32 sequences
Training steps:    260
Training tokens:   199,680 per run
Noise probability: 2.5%
Training period:   4
Shifted period:    6
Optimizer:         AdamW
Learning rate:     0.002
Gradient clip:     1.0
Evaluation:        4 batches of 64 sequences per distribution
Seeds:             0, 1, 2
Hardware:          CPU in the author's benchmark environment
```

The training loss covers all next-token positions. Validation metrics exclude the initial positions whose required motif antecedent has not yet appeared. This is important because the first few outputs are irreducibly uncertain and because the number of such positions differs between period four and period six.

We log negative log-likelihood, perplexity, next-token accuracy, parameter count, elapsed training time, and achieved token throughput.

The principal baseline is ReLU without convolution. We also compare GELU because an architecture improvement that beats only a weak ReLU baseline may look less impressive against a stronger activation.

Finally, we use a longer, 390-step ReLU run as a rough wall-clock comparison. This does not constitute strict hardware-normalized compute matching, but it helps rule out the trivial explanation that Primer simply received much more processing time.

## 15. First ablation: Squared ReLU by itself

Here comes the first result that did **not** reproduce the dramatic effect described in the large-scale paper.

Across three seeds, the small ReLU baseline achieved mean in-distribution negative log-likelihood of **1.615**. Replacing ReLU with Squared ReLU, without adding convolution, gave **1.598**. The GELU baseline scored **1.593**.

Those are small differences.

The corresponding accuracies were **34.3% for ReLU**, **34.8% for Squared ReLU**, and **35.4% for GELU**.

The conclusion is not that Squared ReLU is ineffective. It is that **our tiny 260-step periodic-copy experiment does not provide convincing evidence of a major activation-only benefit**.

This is why reproducing a paper should begin with a faithful statement of what was actually tested. The original work studied larger architectures, longer optimization schedules, different datasets, and different compute regimes. [1][2]

A successful finding at 500 million parameters is not a mathematical guarantee that the same ordering will appear at 50,000 parameters.

## 16. Second ablation: convolution alone changes the picture

The result changes dramatically when we add causal depth-wise convolution to all Q, K, and V projections while keeping ordinary ReLU.

The mean in-distribution negative log-likelihood falls from **1.615** to **0.258**.

Mean next-token accuracy rises from **34.3%** to **95.7%**.

Remember that the model still has ordinary self-attention. The extra operation is not replacing attention with convolutions. It is giving each projected attention vector a learned local temporal filter.

In our task, that is particularly useful. The signal has regular local dependencies and a short repeating motif.

Parameter counts reinforce the point. The original model has **49,248** trainable parameters. The shared-convolution variant has **49,536**. That is an increase of just **288 parameters**, about **0.58%**.

The additional parameters are small, but their placement is powerful.

This does not imply that three-tap convolutions will always transform language-modeling perplexity. It shows that in a setting with strong regular temporal structure, the inductive bias supplied by the convolution can matter much more than its parameter count.

## 17. Primer EZ: the combination works, but which part did the work?

Combining Squared ReLU and the QKV convolutions produced our full Primer-EZ configuration.

Across three seeds, it achieved:

- **0.251** mean in-distribution negative log-likelihood.
- **95.7%** mean in-distribution next-token accuracy.
- **49,536** trainable parameters.
- **3.09 seconds** mean training time for 260 steps on the test CPU.

That is excellent performance on the constructed task.

But compare it with convolution alone: **0.258** negative log-likelihood and **95.7%** accuracy.

The extra gain from Squared ReLU is small here. Nearly all of the difference from vanilla attention is attributable to the convolution.

Our ablations cannot say what percentage of Primer's published large-scale gain comes from each operation. They only isolate effects in this deliberately narrow setting.

The discipline is important. If we reported only the vanilla baseline and full Primer EZ, readers might reasonably conclude that Squared ReLU was a major reason the experiment succeeded. The single-component ablations show that would be misleading.

## 18. The model learned the training pattern. Did it learn to adapt?

Now change one property of the data-generating process.

Instead of repeating a motif every four positions, use a different motif every six positions. Keep the same vocabulary, context length, substitution probability, and model weights. Do not fine-tune.

The results are startling.

The full Primer-EZ model's mean negative log-likelihood increases from **0.251** to **5.883**. Its accuracy falls from **95.7%** to **7.4%**.

The vanilla ReLU model, already much less accurate in-distribution, changes from **1.615** to **2.143** negative log-likelihood. Its shifted accuracy is **21.9%**.

For a 16-symbol vocabulary, uniform guessing gives roughly 6.25% accuracy. Primer's shifted accuracy is close to that chance reference despite its strong performance on familiar sequences.

Why can its shifted negative log-likelihood be so much worse than random guessing? Because a confidently wrong model can assign extremely low probability to the true next token. Cross-entropy punishes those overconfident mistakes much more strongly than ordinary accuracy.

The convolution did not cause a bug. It made it easy to learn a specialized mechanism for the training pattern. When the relationship changed, the model continued applying its learned mechanism.

This is a distribution-shift test, not an argument that Primer EZ generally fails under distribution shift. A motif-length shift is a deliberately adversarial change for a model that has only seen one motif length.

The correct interpretation is narrower and more interesting: **better interpolation can coexist with worse extrapolation**.

## 19. Q-only, K-only, and V-only convolution

The architecture becomes even more informative when we separate the three projections.

Why must all three projections receive the same convolution? The original Primer EZ recipe says to apply it to Q, K, and V because that combination worked well in the researchers' broad experiments. But the three branches play different roles, and our small controlled task lets us study them separately.

We introduced three diagnostic variants:

```python
# Conceptual configuration changes:
q_only = {"use_dconv": True, "conv_targets": "q"}
k_only = {"use_dconv": True, "conv_targets": "k"}
v_only = {"use_dconv": True, "conv_targets": "v"}
```

After 260 steps, the in-distribution mean accuracies were:

- **Q only: 89.0%**.
- **K only: 83.5%**.
- **V only: 74.9%**.
- **All Q, K, V: 95.7%**.

So far, the complete convolution appears strongest.

But the shifted-period experiment changes that ordering:

- **Q only: 31.1%** accuracy.
- **K only: 77.2%** accuracy.
- **V only: 34.3%** accuracy.
- **All Q, K, V: 7.5%** accuracy in the ReLU convolution variant.

The K-only mean shifted negative log-likelihood was **0.762**, compared with **6.049** for convolution on all three branches without Squared ReLU.

That result deserves investigation. One plausible hypothesis is that locally enriched keys let queries match useful motif fragments without forcing the queries and retrieved values to adopt the same specialized positional transformation. In contrast, transforming all three branches may make learning the fixed-period shortcut easier.

That explanation is a **hypothesis**, not an established causal mechanism. To test it properly, we would need attention-map analysis, controlled changes to the convolution kernels, longer sequences, multiple pattern families, and independent held-out shifts. The present ablation establishes the performance difference under this particular protocol. It does not explain that difference definitively.

We should not generalize from this toy result into a recommendation to use K-only convolution in production language models.

## 20. Sharing convolution kernels across heads

We also compared the labml-style shared-kernel variant with an implementation in which every attention head has its own depth-wise convolution kernels.

The per-head model contains **50,400** trainable parameters, compared with **49,536** for the shared-convolution Primer-EZ model.

In-distribution performance is nearly identical: mean negative log-likelihood **0.260** for per-head versus **0.251** for the shared-kernel model, with both approaching **95.8%** accuracy.

Under the period shift, the per-head version does less poorly. Its mean shifted accuracy is **13.7%**, compared with **7.4%** for the shared-kernel Primer-EZ variant. But both are far below the K-only result.

This is a good example of why extra parameters alone are not a reliable predictor of generalization.

More independent filters give heads additional flexibility. That freedom can help or hurt depending on what structure the training task exposes.

## 21. Equal optimizer steps are not equal compute

It is easy to look at our 260-step results and declare that Primer EZ is simply faster.

That is not what the timing shows.

On our CPU environment, 260 steps took an average of **2.08 seconds** for the ReLU Transformer and **3.09 seconds** for Primer EZ. The extra convolution made the individual run about **49% slower** in wall-clock time at a fixed number of optimizer steps.

That result is compatible with the original paper, which explicitly states that the useful modifications increase training step time. [2]

We therefore ran a rough equal-wall-time control: increase the ReLU baseline to **390 steps**. This took a mean **3.30 seconds**, approximately the same order of elapsed time as Primer EZ's 260-step run.

After those extra updates, ReLU improved its in-distribution negative log-likelihood from **1.615** to **1.497**. That remains far above Primer EZ's **0.251**.

The comparison supports a claim about this synthetic task: the better Primer-EZ loss cannot be explained merely by giving the model longer CPU wall time.

It does **not** establish a universal numerical speedup factor. Different kernels, hardware, optimizer implementations, sequence lengths, and batch sizes can change the ratio dramatically. Even for the same model, a fused implementation can change the hardware cost without changing the underlying mathematical operation.

## 22. Accuracy and perplexity measure different kinds of success

In the benchmark report, we save both accuracy and negative log-likelihood.

Accuracy asks whether the highest-scoring token was the correct next token. It treats a barely wrong prediction the same as an extremely confident wrong prediction.

Negative log-likelihood cares about the probability the model assigned to the correct token. If it was only mildly uncertain, the loss is relatively small. If it was confidently wrong, the loss can become very large.

Perplexity is the exponential of mean negative log-likelihood. It can be read as a measure of the model's uncertainty in predicting the next symbol, although its exact interpretation depends on tokenization and evaluation protocol.

This distinction becomes particularly important under distribution shift. A model can move from high accuracy to near-chance accuracy, while its cross-entropy degrades by a much larger factor because its wrong predictions remain highly confident.

For real language models, always report perplexity alongside the dataset, tokenizer, context length, and evaluation procedure. Perplexity numbers from different tokenizations are generally not directly comparable.

## 23. What this benchmark actually establishes

Our lab provides evidence for the following limited statements:

- A straightforward PyTorch implementation of the Primer-EZ operations is correct under the included causal and numerical tests.
- On one small synthetic periodic-copy task, local convolution after QKV projections greatly improves in-distribution next-token prediction.
- The Squared-ReLU-only modification has a small effect in this experiment.
- A change in sequence period creates a substantial failure for the full-convolution variants.
- The K-only diagnostic variant transfers much better on this particular shift.
- Those observations are repeatable across three fixed seeds under the specified CPU setup.

It does not establish that Primer EZ always trains faster, that K-only attention is a superior general architecture, or that any of our models approach the performance of Google's original networks.

We did not train on C4, LM1B, PG19, or a comparable natural-language corpus. We did not run evolutionary architecture search. We did not train a 500-million-parameter Transformer or evaluate production inference kernels.

Our benchmark is a **mechanism exploration**, not an official paper reproduction.

## 24. Why the paper and this lab can disagree

One reason is model scale. Many properties of neural networks change as width, depth, batch size, training tokens, and optimization budgets change. An activation function that helps convergence at hundreds of millions of parameters may show little advantage in a two-layer toy model.

Another is the task distribution. Natural language mixes local syntax, long-range retrieval, rare facts, structural dependencies, and many other patterns. A periodic motif dataset has a much simpler set of possible solutions.

A third reason is what counts as optimization. Our small models may converge to shortcut solutions that are highly useful on the training distribution. A larger dataset with wider variation can reward different mechanisms.

Finally, engineering details matter. Our implementation uses manual attention calculations, ordinary PyTorch grouped convolutions, and a small CPU batch. Google's original experiments used highly optimized accelerator infrastructure.

The original paper also reports limits. Its favorable results primarily target decoder-only autoregressive language modeling, with different comparisons for encoder-based tasks. It should not be treated as a universal recipe for every possible Transformer architecture. [2]

## 25. We tested the most dangerous implementation mistake

Before trusting a causal language model, we need to establish that its outputs do not depend on future input tokens.

Our tests construct two inputs that are identical for their first six positions and different afterward.

```python
tokens = torch.randint(0, 16, (1, 12))
changed = tokens.clone()
changed[:, 6:] = (changed[:, 6:] + 3) % 16

model.eval()
with torch.no_grad():
    before = model(tokens)[:, :6]
    after = model(changed)[:, :6]

assert torch.allclose(before, after, atol=1e-5)
```

We apply this principle at multiple levels:

1. A standalone depth-wise convolution with shared kernels.
2. A standalone depth-wise convolution with separate per-head kernels.
3. Ordinary causal attention.
4. Attention with QKV convolution.
5. The complete language model for every experiment configuration.

We also check kernel-group counts, the exact result of a known three-tap filter, output tensor shapes, gradient finiteness, parameter-count changes, and deterministic dataset generation.

Passing these tests does not prove an implementation is free of all defects. But it addresses the easiest silent failure that can invalidate an autoregressive architecture comparison: future-token leakage.

## 26. Run the complete PyTorch workshop

The companion repository is named `primer_ez_transformer_lab` and is organized as a standalone project.

```text
primer_ez_transformer_lab/
    ARTICLE.md
    README.md
    BENCHMARK_NOTES.md
    requirements.txt
    plot_results.py
    primer_lab/
        __init__.py
        model.py
        data.py
        benchmark.py
        demo.py
    tests/
        test_primer.py
    results/
        benchmark.json
        baseline_extra_steps.json
        ...
    figures/
        01_quality_vs_shift.png
        02_accuracy_vs_shift.png
        03_quality_vs_wall_time.png
```

Create a local virtual environment and install PyTorch:

```bash
cd primer_ez_transformer_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the correctness tests first:

```bash
python -m unittest discover -s tests -v
```

Inspect the causal behavior interactively:

```bash
python -m primer_lab.demo
```

Run the complete three-seed experiment:

```bash
python -m primer_lab.benchmark \
  --seeds 3 \
  --steps 260 \
  --batch-size 32 \
  --eval-batches 4 \
  --configs baseline_relu squared_relu_only dconv_only primer_ez \
            gelu_baseline primer_per_head q_only k_only v_only \
  --output results/reproduced.json
```

If you need to break the experiment into shorter runs, use `--start-seed` with `--seeds 1`. This is how the saved experiment was collected. The saved `results/benchmark.json` contains the merged three-seed data.

Generate the article's original figures:

```bash
python -m pip install matplotlib
python plot_results.py
```

No large pretrained models, cloud APIs, authentication tokens, or proprietary corpora are needed.

## 27. Taking the experiment from synthetic sequences to natural language

If you want to test whether Primer EZ improves a practical language model, the next step is not to increase the motif vocabulary from 16 to 32.

It is to change the evaluation protocol.

First, select a corpus whose licensing and preprocessing are clear. Create document-disjoint training and validation splits. Tokenize all configurations with the same tokenizer and the same special tokens.

Second, define at least two compute budgets. At a fixed token budget, compare optimization and generalization. At an approximately fixed hardware budget, compare quality achieved for the compute consumed.

Third, record both a familiar-domain validation set and a distinct domain or temporal holdout. For instance, if your training material is mostly technical prose, evaluate separately on narrative prose or on a newer collection not used for training.

Fourth, ensure that the only architectural changes are those being studied. Keep model widths, depth, optimizer, learning-rate schedule, token count, data ordering, context length, and initialization protocol consistent.

Finally, measure uncertainty across multiple seeds. A difference of 0.01 in cross-entropy may or may not be practically meaningful depending on data variance and evaluation size.

A model comparison is not a leaderboard entry until its measurement conditions can be reproduced.

## 28. What a GPU benchmark should change

The tutorial's reference implementation computes attention with ordinary matrix multiplication and explicit masking. This is valuable pedagogically but is not the most efficient path for a modern GPU.

When moving to GPU, use PyTorch's `scaled_dot_product_attention` or an appropriate optimized attention backend, subject to the causal-mask and dtype requirements documented for that implementation. [7]

Keep the Q, K, and V convolutions before the attention call. Test the optimized path against the reference implementation on short sequences before benchmarking.

Then profile:

- Convolution kernel-launch overhead.
- Attention time and memory.
- Tensor reshapes and materialization.
- Mixed-precision numerical stability.
- Throughput as context length increases.
- Prefill and token-by-token decoding separately.

Do not assume a training-time improvement implies an inference-time improvement. During autoregressive decoding, a width-three convolution needs the two most recent projected vectors. A naive implementation that repeatedly convolves the entire context could be unnecessarily expensive.

A more serious inference implementation would maintain a short per-layer state cache for the convolution, alongside attention's key-value cache. That is an engineering extension of the causal convolution idea, not something implemented or benchmarked by our CPU lab.

## 29. The experiments I would run next

There are several natural follow-ups if the goal is to turn this into a strong research project rather than a demonstration.

**First, vary the training-period distribution.** Instead of training only on period four, train on periods two through six. Test on period seven and eight. This reveals whether the current result reflects architectural generalization or simply exposure to a single recurring distance.

**Second, perform a context-length sweep.** Increase the sequence from 24 to 48, 96, and 192 tokens. Keep a controlled parameter budget and record how the quality-compute tradeoff changes.

**Third, measure attention maps and learned convolution weights.** Does QKV convolution create heads that specialize in fixed lags? Does K-only convolution preserve more flexible retrieval? The present loss numbers suggest hypotheses but cannot answer those questions alone.

**Fourth, test multiple seed families and held-out pattern generators.** A single synthetic grammar is useful for debugging but insufficient for architectural conclusions.

**Fifth, add a strong modern Transformer baseline.** The original paper compared against alternatives more sophisticated than ordinary ReLU. A current study should include GELU, gated FFNs such as SwiGLU, and fused attention where appropriate.

**Sixth, report quality as a function of measured accelerator time or FLOPs.** A fixed number of steps is valuable for an ablation but not enough to establish training-compute efficiency.

**Seventh, distinguish training optimization from deployment cost.** A design that improves loss but complicates efficient autoregressive decoding might still be worthwhile, but the tradeoff should be explicit.

**Eighth, test the full official Primer architecture separately from Primer EZ.** Simplifying an architecture is itself an experiment. It deserves its own controls.

## 30. What Primer teaches us about architecture search

The enduring value of Primer is not the claim that a three-position convolution should be included in every future language model.

It is the evidence that surprisingly small numerical operations can change a model's learning behavior enough to matter at scale.

The original Transformer was a conceptual breakthrough because it showed that attention could replace much of the recurrence and convolution used in previous sequence models. Primer did not overturn that conclusion. It showed that attention itself can benefit from a little local computation at the right point in the graph.

That is a recurring theme in machine learning architecture research: the most valuable changes do not always look impressive in a block diagram.

A different nonlinearity can change optimization. A tiny convolution can change which relationships are easy to represent. A parameter-sharing decision can change generalization. None of these requires a fundamentally new modeling paradigm.

At the same time, our independent experiment demonstrates why architecture research must be tied to honest evaluation. The model that learned our familiar repeating patterns best failed badly after the pattern changed. The model that generalized best used a particular subset of the convolutions, not the complete configuration we initially expected to win.

There is no contradiction here. **An inductive bias is a bet about structure.** When the data contains that structure, the model learns efficiently. When the structure changes, the same bias can become a liability.

This is the practical lesson to carry from Primer EZ into modern model design:

**Do not ask only whether an architectural modification helps. Ask what it makes easy to learn, what it makes difficult to learn, and which measurement would reveal the difference.**

---

## 31. References and further reading

**[1]** So, D. R., Mańke, W., Liu, H., Dai, Z., Shazeer, N., and Le, Q. V. (2021). *Primer: Searching for Efficient Transformers for Language Modeling*. NeurIPS 2021. [NeurIPS publication](https://papers.nips.cc/paper_files/paper/2021/hash/2f3c6a4cd8af177f6456e7e51a916ff3-Abstract.html) | [arXiv](https://arxiv.org/abs/2109.08668).

**[2]** So et al. *Original Primer paper, full methods and appendices*. [Research PDF](https://papers.nips.cc/paper/2021/file/2f3c6a4cd8af177f6456e7e51a916ff3-Paper.pdf). Details the evolutionary program search, causal primitive operations, ablations, and large-scale quality-to-compute comparisons.

**[3]** labml.ai. *Primer EZ: annotated PyTorch implementation*. [Read the implementation](https://nn.labml.ai/transformers/primer_ez/index.html).

**[4]** labml.ai. *Primer EZ source code*. [GitHub](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/primer_ez/__init__.py).

**[5]** Shazeer, N. (2020). *GLU Variants Improve Transformer*. [arXiv](https://arxiv.org/abs/2002.05202). Useful context for multiplicative feed-forward nonlinearities.

**[6]** PyTorch. *torch.nn.Conv1d*. [Official documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html). Documents grouped and depth-wise convolution semantics.

**[7]** PyTorch. *scaled_dot_product_attention*. [Official documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention). Documents causal attention and optimized execution.

**[8]** Vaswani, A., et al. (2017). *Attention Is All You Need*. [arXiv](https://arxiv.org/abs/1706.03762). Foundational Transformer architecture.

**[9]** Google Research. *Primer publication page*. [Google Research](https://research.google/pubs/primer-searching-for-efficient-transformers-for-language-modeling/).

**[10]** Google Research. *Primer reference project*. [Original Google repository](https://github.com/google-research/google-research/tree/master/primer). The project's maintained availability and setup requirements should be verified before claiming a full-scale replication.

---

*The companion `primer_ez_transformer_lab` project contains independent PyTorch code, causal-leakage tests, three-seed benchmark results, quality-versus-compute plots, and methodology notes. The original paper's published results and this article's tiny-model measurements are deliberately kept separate.*
