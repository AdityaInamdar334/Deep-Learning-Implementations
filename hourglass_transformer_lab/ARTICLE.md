# A Transformer Learned to Think in Fewer Tokens. The Hard Part Was Not Leaking the Future.

### Inside Hourglass Transformers: hierarchical attention, causal downsampling, learned resampling, and a complete PyTorch lab that tests what happens when the sequence changes

*Based on "Hierarchical Transformers Are More Efficient Language Models" by Piotr Nawrot and colleagues, Findings of NAACL 2022, and the annotated implementation by labml.ai.*

---

Imagine a language model that processes twelve tokens. A regular Transformer carries twelve representations through every layer. An Hourglass Transformer might preserve all twelve in its first layer, compress them to four, perform much of its computation on those four, and then expand back to twelve before predicting the next token.

For a sufficiently long sequence, this is a compelling way to spend less computation on redundant information. A full attention operation on twelve tokens considers 144 query-key pairs. On four tokens, it considers sixteen. That is nine times fewer pairs in the shortened attention layer.

But there is a problem that can make the entire experiment scientifically invalid.

Suppose our model compresses tokens 0, 1, and 2 into one vector, then broadcasts that vector back to positions 0, 1, and 2. The representation at position 0 now contains information from tokens 1 and 2. If it is asked to predict token 1, it has already seen the answer.

Every attention mask can be perfectly triangular. Every training loss can decrease. The model can look outstanding on its benchmark. It is still cheating.

This is what makes the Hourglass Transformer an exceptionally interesting architectural case study. Its central difficulty is not reducing the number of tokens. It is **reducing tokens without violating the meaning of autoregressive prediction**.

The original paper solved this with a surprisingly small operation: before reducing the sequence by a factor of `k`, shift the representations right by `k - 1` positions. Combine that with token-resolution Transformer layers, carefully designed upsampling, and residual connections, and you have an architecture that can spend most of its depth on shortened sequences while retaining valid next-token probabilities. [1]

In this tutorial, we will build and test that idea from scratch.

We will also discover a second lesson. In our independent toy benchmark, learned linear resampling substantially improves fitting to a familiar repeating-token pattern. The very same model almost completely loses that advantage when the repeat period changes. Efficiency and generalization are separate claims, and both need experiments.

**What we built:** An independent CPU-friendly PyTorch implementation with vanilla and hierarchical models, average and learned linear pooling, recursive shortening, variable shortening factors, intentional leakage controls, reproducible multi-seed experiments, and 25 correctness tests. It is a teaching implementation, not the authors' official model or a reproduction of their enwik8 or ImageNet experiments.

---

## Table of contents

1. The research question hidden inside a U-shaped network
2. What the authors actually demonstrated
3. Why full-resolution attention is expensive
4. The Hourglass architecture, one layer at a time
5. Three different kinds of information flow
6. The causal pooling trap
7. Proving why the extra shift is `k - 1`
8. Building the shift operation in PyTorch
9. Implementing average pooling
10. Reconstructing token resolution
11. The skip connection is doing real work
12. Building a minimal causal Transformer block
13. Combining the blocks into an Hourglass
14. Recursive shortening and nested hierarchies
15. Learned linear pooling and upsampling
16. Attention-based resampling in the original paper
17. Why fine-resolution Transformer layers matter
18. Parameter count versus actual computation
19. The synthetic language-modeling experiment
20. The evaluation protocol
21. What the baseline learned
22. The shortened variants
23. Why learned pooling looked so strong
24. Distribution shift exposes a different problem
25. Testing longer sequences
26. The deliberately broken, unshifted model
27. Ablating skip connections
28. Varying the shortening factor during training
29. Are the models actually faster?
30. The paper's published language and image results
31. What the paper's other ablations reveal
32. Why a fast small model is not a faithful reproduction
33. Practical failure modes and engineering safeguards
34. Extending the lab to real data
35. Open research directions
36. Reproducibility and references

---

## 1. The research question hidden inside a U-shaped network

In 2021, Piotr Nawrot, Szymon Tworkowski, Michał Tyrolski, Łukasz Kaiser, Yuhuai Wu, Christian Szegedy, and Henryk Michalewski asked a question that sounds almost embarrassingly obvious in retrospect:

Why must every layer of a language model operate on every token at the original sequence resolution?

Consider a paragraph describing a flight delay. Some operations need exact words and punctuation. Others need much broader information: who is flying, when the delay happened, and why it matters. Carrying every character through every layer may not be the most efficient use of a network's capacity.

This distinction resembles image processing. A U-Net progressively reduces spatial resolution to build higher-level features, then recovers fine detail while reusing information from earlier, high-resolution layers. Hourglass takes this broad idea into autoregressive sequence modeling. [1, 6]

The model begins with a full-resolution representation, descends to a shorter representation, performs substantial processing there, and ascends to full resolution for prediction.

But language is not an image segmentation task. During next-token prediction, position `t` may use the current and previous input tokens but cannot see a future input. Any shortening operation that mixes future tokens into earlier output positions violates the probability factorization the model is supposedly learning.

This is the central technical challenge. A U-shaped network is the easy part. A *causal* U-shaped network is the interesting part.

## 2. What the authors actually demonstrated

The original Hourglass paper was first submitted in October 2021 and appeared in Findings of NAACL 2022. Its contributions are more specific than the broad claim that hierarchy makes everything efficient. [1]

The researchers explored different ways to shorten and reconstruct sequences, described the extra right shift needed for autoregressive safety, evaluated alternative layer distributions, and showed that additional Transformer layers at shortened resolutions can improve model quality for a given computational budget.

They tested Hourglass with both standard relative attention in a Transformer-XL-style backbone and the more efficient locality-sensitive hashing attention used by Reformer. This matters because Hourglass is not itself a replacement for attention. It is a way to decide **at which sequence resolution attention should operate**. [1, 2, 5]

They also introduced *shorten factor dropout*, which varies the shortening factor during training. The goal is to prevent the model from depending too heavily on one fixed grouping of neighboring tokens.

The reported gains are meaningful but not universal. The experiments used specific datasets, optimization schedules, attention implementations, compute resources, and resampling choices. A tiny PyTorch model on synthetic sequences cannot inherit those results by association.

Our experiment therefore asks a narrower question: do the architectural mechanisms work as intended, and what tradeoffs can we directly measure?

## 3. Why full-resolution attention is expensive

Start with a sequence of length `N` and hidden width `D`.

A conventional attention layer constructs queries, keys, and values for all `N` tokens. The attention scores compare every query with every available key. Ignoring causal masking's constant factor, the attention-score work scales with `N * N`.

If we reduce the sequence to `N / k` tokens, attention scores within that shortened layer scale with:

`(N / k) * (N / k) = N² / k²`

For `k = 4`, a shortened attention layer operates on about one-sixteenth as many query-key pairs. The feed-forward part also sees one-quarter as many tokens.

Crucially, **this is not a sixteen-fold speedup of the entire model**. Projection layers, embedding lookups, residual paths, the full-resolution layers, pooling, upsampling, memory traffic, and software overhead still exist.

For example, consider the four-layer networks in our companion lab:

- Vanilla: four attention blocks on 48 tokens each.
- Hourglass: one block on 48 tokens, two on the shortened sequence, and one more on 48 tokens.

With `k = 3`, the vanilla network has a simple attention-pair proxy of `4 * 48² = 9,216`. The Hourglass has `2 * 48² + 2 * 16² = 5,120`. That is roughly 44% fewer nominal attention pairs across those blocks, not 89% fewer operations across the whole program.

And this accounting does not establish memory savings, GPU throughput, inference speed, or energy use. Those are separate measurements.

## 4. The Hourglass architecture, one layer at a time

The architecture can be written as a short sequence of operations:

```text
Full-resolution token embeddings
              |
     Full-resolution block
              |
      Shift right by k - 1
              |
      Shorten by factor k
              |
      Short-resolution blocks
              |
       Expand by factor k
              |
  + Full-resolution skip path
              |
     Full-resolution block
              |
       Next-token logits
```

What makes this structure an hourglass is the shrinking and expanding token axis. The hidden width can remain unchanged while the sequence length changes.

For an input tensor shaped `(batch, 48, 32)`, shortening by three produces `(batch, 16, 32)`. The middle blocks therefore process sixteen representations rather than forty-eight. Upsampling restores `(batch, 48, 32)` so the model can emit one distribution over the vocabulary at each original position.

In the original paper, the architecture is expressed recursively: each shortened stage can itself contain another Hourglass stage. This makes configurations with multiple levels possible, such as 48 tokens becoming 24, then 12, then returning to 24 and 48. [1]

Our PyTorch implementation supports both a single shortening factor and nested factors. The main benchmark keeps a single level so the ablations remain interpretable.

## 5. Three different kinds of information flow

It helps to separate three concepts that often get collapsed into one diagram.

**Token-level causal attention** answers the question: which previous token representations can the current token attend to?

**Shortening** answers a different question: which neighboring fine-resolution representations are merged into one coarser representation?

**Upsampling** answers a third: which fine-resolution output positions receive information from each coarse representation?

Only the first of these is controlled by the ordinary triangular attention mask. A model can enforce perfect causal attention within every block and still leak future information through the other two operations.

That is why simply passing `is_causal=True` to a PyTorch attention function is insufficient.

## 6. The causal pooling trap

Consider the input representations at positions 0 through 11, with shortening factor three.

Without any extra shift, average pooling groups them like this:

```text
Original positions
[ 0  1  2 ] [ 3  4  5 ] [ 6  7  8 ] [ 9 10 11 ]
      |             |             |             |
   coarse 0      coarse 1      coarse 2      coarse 3
      |             |             |             |
[ 0  1  2 ] [ 3  4  5 ] [ 6  7  8 ] [ 9 10 11 ]
Reconstructed positions
```

Every reconstructed position in the first group can now depend on original inputs 0, 1, and 2. That includes reconstructed position 0.

If the model produces the prediction of the token following position 0, it has already incorporated input token 1. The model has seen precisely what it was supposed to predict.

Worse, this is not merely a training artifact. The model cannot legally produce the same distribution during generation when future tokens are unavailable. A benchmark that permits this path is testing an invalid predictor.

The paper identifies this issue explicitly and introduces a separate right shift before each shortening operation. [1]

## 7. Proving why the extra shift is `k - 1`

Take a coarse group `g`. It will eventually be repeated into fine-resolution positions `g*k` through `g*k+k-1`.

For the group to be safe at its *earliest* output position `g*k`, it must not depend on any input position beyond `g*k`.

Without shifting, the pooled group contains input positions up to `g*k+k-1`. That is too late.

Now shift the fine-resolution sequence right by `k-1` before pooling. The source indices entering coarse group `g` become:

`g*k - (k-1)` through `g*k`

The latest source position is exactly `g*k`, which is the first fine-resolution position receiving that coarse group.

Therefore, every recipient position in the group can use the coarse representation without seeing a future original input.

For `k = 3`, the safe grouping becomes:

```text
Shifted inputs
[ 0  0  x0 ] [ x1  x2  x3 ] [ x4  x5  x6 ] ...
       |               |               |
    coarse 0        coarse 1        coarse 2
       |               |               |
[ out0 out1 out2 ] [ out3 out4 out5 ] [ out6 out7 out8 ]
```

For prediction at output position 3, `coarse 1` can depend on inputs 1, 2, and 3. None is in the future of position 3. For output position 5, that same coarse information is older, which is also safe.

The proof assumes that the operations *inside* the coarse stage are causal as well. The shift protects the pooling boundary; the attention mask protects operations within each resolution.

One detail deserves emphasis: an autoregressive language-model training loop already shifts **targets** by one token. That normal next-token alignment is distinct from the additional `k-1` shift applied to the *hidden representations before pooling*. You need both.

## 8. Building the shift operation in PyTorch

In the companion repository, tensors use the batch-first convention: `(B, T, D)`. The original labml.ai annotated code uses sequence-first tensors: `(T, B, D)`. The logic is equivalent, but we keep batch-first because it integrates naturally with modern PyTorch modules.

Here is the essential implementation:

```python
import torch


def causal_shift(x: torch.Tensor, amount: int) -> torch.Tensor:
    if amount < 0:
        raise ValueError("Shift must be nonnegative")
    if amount == 0:
        return x
    if amount >= x.shape[1]:
        return torch.zeros_like(x)

    prefix = x.new_zeros(x.shape[0], amount, x.shape[2])
    return torch.cat((prefix, x[:, :-amount]), dim=1)
```

For example:

```python
x = torch.arange(1, 7).float().view(1, 6, 1)
print(causal_shift(x, 2).flatten())
# tensor([0., 0., 1., 2., 3., 4.])
```

The zeros are not future information. They act as a fixed, learned-around initial context. We do not use `torch.roll`, because rolling would move tokens from the end of the sequence into the beginning, instantly violating causality.

The repository tests negative shifts, zero shifts, shifts larger than the sequence, and the precise first positions for known examples.

## 9. Implementing average pooling

Average pooling is the simplest shortening mechanism: take every group of `k` consecutive representations and average them.

```python
import torch.nn.functional as F


def average_shortening(x, k):
    # x: [batch, sequence, hidden]
    return F.avg_pool1d(
        x.transpose(1, 2),
        kernel_size=k,
        stride=k,
        ceil_mode=True,
    ).transpose(1, 2)
```

The transposition matters. `avg_pool1d` expects the sequence along its last dimension, not the middle one.

The `ceil_mode=True` argument keeps a final partial group when the sequence length is not divisible by the factor. In the lab, we explicitly test lengths such as 25 with factors of three or four.

This method is inexpensive, parameter-free, and easy to verify. It is not necessarily expressive enough for token-level language modeling. Two very different neighborhoods can have similar means, and simple averaging cannot learn which parts of the neighborhood matter most.

That is one reason the original researchers also studied learned resampling. [1]

## 10. Reconstructing token resolution

The most basic upsampling method repeats every coarse representation `k` times:

```python
def repeat_upsample(x_short, k, original_length):
    expanded = torch.repeat_interleave(x_short, k, dim=1)
    return expanded[:, :original_length]
```

For a sequence shortened from twelve positions to four with `k=3`, this returns twelve positions again. All three positions within each group receive the same coarse vector.

This does not mean all twelve final representations must become identical. In the full architecture, the repeated vector is added to a high-resolution skip connection and then processed by another full-resolution Transformer layer.

But the coarsened branch alone cannot distinguish the three positions within a group. That limitation matters for detailed token prediction.

## 11. The skip connection is doing real work

The hourglass's upward path reconnects with the representation from before shortening:

```python
expanded = repeat_upsample(shortened, k, dense.shape[1])
x = dense + expanded
```

This is not a decorative residual connection.

The coarse representation can encode broader or older context. The dense representation retains the distinctions that compression might discard, including token identity, precise position, and local features.

Without the skip, the model's final full-resolution block must recover fine-grained token information from repeated summaries and positional signals. Sometimes it can. Often the task becomes unnecessarily difficult.

The original Hourglass paper follows the same broad design principle that makes encoder-decoder skip connections useful in U-Net, although the application and causality constraints differ. [1, 6]

Our lab includes an explicit `hourglass_no_skip` ablation so readers can see how the score changes when this path is removed.

## 12. Building a minimal causal Transformer block

We need a standard Transformer baseline and the full-resolution and shortened blocks inside Hourglass. For comparability, we use the same implementation in both places.

Modern PyTorch exposes scaled dot-product attention, which can use optimized kernels depending on the device and input conditions:

```python
import torch.nn.functional as F


def attention(q, k, v):
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
    )
```

The real block wraps this operation with learned QKV projections, an output projection, LayerNorm, a feed-forward network, and residual connections.

Our module looks like this at a conceptual level:

```python
class TinyCausalBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.heads = heads

    # The companion source includes the complete tensor reshaping
    # and calls scaled_dot_product_attention with is_causal=True.
```

This is enough structure to test the architectural hypothesis without importing large language-model libraries. The full implementation, including the QKV/head dimension reshaping, is in `hourglass_lab/models.py`.

The test suite checks that earlier outputs do not change if a future input is modified. This invariant matters more than matching a schematic diagram.

## 13. Combining the blocks into an Hourglass

The core forward pass is compact:

```python
def forward(self, x, override_factor=None):
    k = override_factor or self.factor

    dense = self.pre(x)

    # Additional shift is essential for causal downsampling.
    shifted = causal_shift(dense, k - 1)
    short = self.down(shifted, k)

    short = self.inner(short)
    expanded = self.upsample(short, k, dense.shape[1])

    x = dense + expanded
    return self.post(x)
```

The actual project supports switching the shift off, switching the skip off, and swapping average resampling for learned linear resampling. This is deliberate. If a research implementation cannot disable or replace its core mechanism, it is harder to construct meaningful ablations.

An embedding layer and a next-token prediction head wrap the core. Training still uses ordinary cross-entropy over next-token targets.

```python
sequence = make_batch(...)      # [B, T + 1]
inputs = sequence[:, :-1]       # [B, T]
targets = sequence[:, 1:]       # [B, T]

logits = model(inputs)          # [B, T, vocab]
loss = F.cross_entropy(
    logits.reshape(-1, vocab),
    targets.reshape(-1),
)
```

Notice that the model does not predict a single compressed symbol for each group. It still predicts **one next-token distribution per original time step**.

## 14. Recursive shortening and nested hierarchies

A genuine hierarchical architecture may have more than one level.

```text
48 original tokens
      |
24 coarse tokens       k = 2
      |
12 coarser tokens     k = 2
      |
24 coarse tokens
      |
48 original tokens
```

The companion project accepts `factors=(2, 2)`. Each recursive level contains its own causal pre-block, its own shift before shortening, a shorter inner stage, an upward reconstruction, and a causal post-block.

Nested shortening does not magically preserve causality. The property is preserved because each level individually satisfies it, and because a composition of causal functions remains causal.

The tests verify a nested configuration using non-divisible input lengths. A successful shape test alone would not be enough. The suite also perturbs future inputs and checks earlier logits.

Our main quantitative comparison uses one level because a nested model has extra blocks and is not naturally matched to the four-block baseline in depth.

## 15. Learned linear pooling and upsampling

Average pooling does not distinguish which input pattern created a mean. Learned linear pooling can.

First group `k` neighboring vectors, concatenate them, and project back to hidden width `D`:

```python
class LinearShortener(nn.Module):
    def __init__(self, width, factor):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(width * factor, width)

    def forward(self, x, factor):
        if factor != self.factor:
            raise ValueError("Fixed factor required")
        batch, length, width = x.shape
        pad = (-length) % factor
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        grouped = x.reshape(batch, (length + pad) // factor,
                            width * factor)
        return self.proj(grouped)
```

Unlike average pooling, this operator can learn a different coefficient for each position inside a group. It can choose to emphasize the most informative location and suppress others.

The reverse operation can project a coarse vector from `D` dimensions to `k*D` dimensions and reshape it into `k` separately parameterized output vectors.

```python
class LinearUpsampler(nn.Module):
    def __init__(self, width, factor):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(width, width * factor)

    def forward(self, short, original_length):
        b, groups, d = short.shape
        dense = self.proj(short).reshape(b, groups * self.factor, d)
        return dense[:, :original_length]
```

The causal shift remains necessary. A learnable pooling matrix does not acquire a guarantee against leakage just because its weights are trained.

There is also a cost: linear resampling introduces additional parameters, requires fixed factor-specific weight shapes, and may specialize to the local structure of the training distribution.

That last effect becomes the most interesting result in our independent experiment.

## 16. Attention-based resampling in the original paper

The authors did not stop at averaging or linear projection. They explored shortening and upsampling with attention.

In attention-based shortening, a preliminary short sequence provides queries, while the higher-resolution sequence provides keys and values. The shortened representations can selectively extract useful information from the input, rather than being forced to use an average.

Attention-based upsampling works in the reverse direction. Fine-resolution queries retrieve information from the shortened representations, with the goal of reconstructing relevant group-level information at each position.

The published results generally favor the stronger attention-based upsampling choices. The authors' actual large-scale setups therefore have more expressivity than the minimal annotated labml.ai code. [1, 3]

Our local benchmark implements average and linear resampling because they are easy to audit, inexpensive on CPU, and expose a clean distinction between a fixed reduction rule and a learned one. It does **not** claim to implement every attention-based resampling variant from the paper.

For an extension, the critical design question is the mask used in cross-resolution attention. A coarse query at output group `g` must not attend to fine-resolution keys later than its earliest output position. The same dependency proof used for average pooling should be applied to the learned attention operator.

## 17. Why fine-resolution Transformer layers matter

Could we just shorten immediately and do all computation on the coarse sequence? Not without paying a cost.

Suppose one coarse token represents three original tokens. With no fine-resolution path, the model may be unable to represent dependencies *within* that group at the level required for next-token prediction.

Even with safe pooling, the same coarse state is used for multiple output positions. Those positions need different conditional distributions, because each has access to different prefixes.

The pre-block lets fine-resolution tokens extract useful local information before shortening. The post-block combines the coarse context with the upsampled fine-resolution path and can distinguish positions again.

The paper explicitly studies this issue. Its authors find that removing full-resolution layers harms language modeling, and that a balanced distribution of pre- and post-blocks is effective in their tested configurations. [1]

The important principle is not that one particular block count is universally best. It is that **compression saves compute by discarding distinctions, while autoregressive prediction still depends on some of those distinctions**.

## 18. Parameter count versus actual computation

Here is a trap in neural architecture comparisons: smaller attention matrices do not imply fewer parameters.

A Transformer block uses matrices whose dimensions mostly depend on `D`, not the sequence length `N`. Moving that block from 48 tokens to 16 tokens does not delete its learned QKV or feed-forward matrices.

In our baseline experiment:

- The four-block vanilla model has **36,320 parameters**.
- The average-pooling Hourglass variants also have **36,320 parameters**.
- The learned linear-resampling variant has **42,592 parameters**.

Therefore, the equal-parameter Hourglass comparison isolates sequence resolution more cleanly than the learned resampling comparison does.

Linear pooling adds 6,272 trainable parameters. Its superior score may reflect better inductive bias, additional capacity, optimization differences, or a combination. The small benchmark is not sufficient to separate all those explanations.

In large models, Hourglass can place more layers at coarse resolution and grow parameter count while retaining similar training cost. The original paper demonstrates precisely why parameter count and compute become weakly coupled in hierarchical networks. [1]

## 19. The synthetic language-modeling experiment

We want an experiment small enough to reproduce on a laptop, but not so trivial that every model gets the same score immediately.

We construct sequences of integer tokens from a vocabulary of 32 symbols. Each sequence begins with four random tokens. Most subsequent tokens repeat the symbol four positions earlier, with a small amount of random corruption.

The generator follows a simple rule:

```python
for t in range(period, length + 1):
    previous = sequence[t - period]
    noise = random_token()
    sequence[t] = noise if random_perturbation() else previous
```

Our training regime uses a period of four and a token corruption probability of 7%. This creates a learnable recurring dependency. The model must recognize the relevant previous position and predict which symbol is likely to recur.

We then create three independently seeded evaluation sets:

1. **Familiar distribution:** the same period of four, with newly sampled sequences.
2. **Period shift:** the copying rule changes to seven tokens earlier.
3. **Length shift:** the original period remains four, but the sequence length grows from 48 to 64.

The last two tests probe different abilities. Extending the length preserves the underlying rule. Changing the period changes the rule itself. An architecture can succeed at one without succeeding at the other.

This is intentionally not natural language. There is no tokenization, grammar, semantics, multilingual transfer, or human evaluation. The point is to investigate a controlled inductive bias, not to demonstrate general language-model performance.

## 20. The evaluation protocol

We train eight configurations. For each configuration, we run seeds 0, 1, and 2. Each model receives **95 optimizer steps** using batch size 12, a sequence of 48 input tokens, a hidden width of 32, and four attention heads.

The core variants are:

```text
vanilla_4
hourglass_k2
hourglass_k3
hourglass_k4
hourglass_linear_k3
hourglass_unshifted_UNSAFE
hourglass_no_skip
hourglass_factor_dropout
```

All models use the same next-token cross-entropy objective, optimizer family, vocabulary, train seed protocol, evaluation sets, and broad Transformer block implementation. They do not all have equal parameter counts or total floating-point operations.

We measure:

- Cross-entropy in nats and perplexity across all next-token positions.
- Copy-region accuracy at positions where the recurrence rule is established.
- Training time measured on the local CPU, not inferred from attention complexity.
- Model parameter counts.
- Future-token leakage, measured by whether editing a later input changes earlier output logits.

The evaluator uses independent random seeds for the training and validation sequences. The result files contain individual trial metrics and aggregate mean and sample standard deviation.

The measured output is saved as `results/benchmark.json`. Figures are generated from that file, not copied from the paper.

## 21. What the baseline learned

The vanilla four-block Transformer reached **13.3%** mean copy-region next-token accuracy on familiar sequences after the 95-step budget. On the changed-period evaluation it reached **8.9%**.

Those numbers are well above the 3.125% chance rate of choosing among 32 uniformly distributed tokens, but they are far below a nearly perfect copy mechanism.

This is worth taking seriously. The baseline is not a fully optimized standard Transformer; it is a deliberately tiny model trained for a short budget. The results answer a question about early training and inductive bias, not final attainable perplexity.

Its mean measured training time was approximately **1.195 seconds** for 95 steps on the CPU environment used for this run.

Now we can compare what happens when the same parameter budget is reorganized into fine and coarse processing stages.

## 22. The shortened variants

All three average-pooling Hourglass variants shared the vanilla model's 36,320 trainable parameters.

On the familiar recurrence task, mean copy-region accuracies were:

- **Hourglass, k=2:** 22.1%.
- **Hourglass, k=3:** 15.6%.
- **Hourglass, k=4:** 17.9%.
- **Vanilla, four layers:** 13.3%.

Every average-pooling Hourglass variant improved the familiar score over the standard Transformer under this particular short training budget, although the differences were much smaller than for the learned pooling model.

The ranking did not increase monotonically with the shortening factor. In fact, `k=2` performed better than both `k=3` and `k=4`.

This is a useful reminder that shortening factor is not just a compute knob. It changes where information is grouped, how much detail is lost, and which recurrences align with group boundaries. A larger `k` reduces attention pair counts more aggressively but does not necessarily improve learning.

## 23. Why learned pooling looked so strong

The strongest familiar-distribution result came from `hourglass_linear_k3`.

Across three seeds, it reached **55.8%** mean accuracy after 95 steps, compared with 13.3% for vanilla and 15.6% for average-pooling `k=3`.

Why might learned pooling help so much?

Averaging three vectors gives the model an undifferentiated summary. A learned linear projection can assign distinct roles to the first, second, and third positions inside each group. Repeated-token patterns are highly structured. A small learned resampler can align its projections with those stable positions, while average pooling must preserve all positional information indirectly in the hidden representation.

The learned upsampler also gives each reconstructed position a different projection, rather than repeating identical group-level vectors.

But another interpretation is possible: the model simply has more parameters and potentially more favorable optimization geometry.

The proper next experiment would match additional parameters in the vanilla Transformer or average-pooling Hourglass and compare at equal FLOPs or equal wall-clock budgets. Our current lab intentionally leaves this question open rather than attributing all the improvement to one mechanism.

The next test reveals a more serious limitation.

## 24. Distribution shift exposes a different problem

When we change the underlying recurrence from period four to period seven, the learned linear-pooling Hourglass falls from **55.8% to 4.4% accuracy**.

The conventional Transformer falls from 13.3% to 8.9%. The average-pooling `k=2` model falls from 22.1% to 10.0%.

In relative terms, the learned resampling model suffers the most dramatic collapse. The mechanism that made it exceptionally good at a stable periodic pattern is much less helpful when the location of the relevant symbol changes.

This experiment does not prove that learned pooling is inherently less robust on natural language. It demonstrates that **a learnable compression operator can encode a useful, narrow positional shortcut**.

The lesson generalizes to architecture evaluation: a strong in-distribution metric may reflect genuine modeling ability, shortcut learning, or both. If we want to claim a more general improvement, we need to evaluate where the relevant evidence moves.

For future work, we could train on a mixture of periods, randomize where repeated motifs begin, or construct tasks that require local syntax and long-range semantics simultaneously. Those are more demanding tests of the claimed multiscale benefit.

## 25. Testing longer sequences

Changing sequence length while keeping the recurrence fixed produces a different outcome.

The linear-pooling `k=3` model reaches **54.6%** on sequences of length 64 after training only on length 48. This is close to its 55.8% familiar score.

So the model is not simply incapable of processing an unfamiliar number of tokens. The architecture has no fixed learned token-position table in the main body, and we use sinusoidal position embeddings, so a longer input remains structurally valid.

The failure under period shift comes from changing what the tokens mean in relation to one another, not from changing the raw length.

It is important not to exaggerate this result. We have shown extrapolation from 48 to 64 tokens for one artificial rule. We have not shown million-token memory, long-document coherence, or a production streaming decoder.

## 26. The deliberately broken, unshifted model

The control named `hourglass_unshifted_UNSAFE` disables the extra right shift before pooling.

Its attention layers remain individually causal. Nevertheless, the pooled representation can use future input tokens within a group. That contaminated information is then broadcast to earlier output positions.

Our leakage check creates an input sequence, changes a later token, and measures the maximum absolute difference in logits at earlier positions. For valid models, the measured difference is zero to numerical precision. For the unshifted model, the average maximum difference is approximately **0.760** across three seeds.

That is not a tiny statistical fluctuation. It is a direct failure of the autoregressive dependency contract.

Here is the diagnostic pattern:

```python
model.eval()
with torch.no_grad():
    original = model(tokens)
    changed_tokens = tokens.clone()
    changed_tokens[:, 5] = 17
    changed = model(changed_tokens)

# Earlier positions must not know about the change at position 5.
leak = (original[:, :5] - changed[:, :5]).abs().max()
assert leak < 2e-5
```

The exact test code changes several positions across several model configurations and uses distinct values to ensure an actual intervention occurred.

**Never compare the unshifted model's validation loss against a correctly causal model as if they were equivalent language models.** The invalid model has access to information that would not exist at generation time.

This is why a good benchmark needs more than loss curves. It needs tests of what information the model was allowed to use.

## 27. Ablating skip connections

Removing the skip connection produced an interesting small-model result. The no-skip model reached **24.0%** familiar-distribution accuracy, higher than the corresponding average-pooling `k=3` model's 15.6%.

That is not proof that skip connections are useless. In a short optimization budget, removing a residual path can change scale, gradient flow, and the relative influence of shortcut features. These effects may temporarily help one synthetic task.

Under period shift, the no-skip model fell to **4.3%**, below the average-pooling `k=3` model's 5.2%.

This illustrates why ablations should include more than the original training distribution and why it is dangerous to turn one short-run result into a universal design rule.

A more complete skip-connection study would use longer training, equal wall time, several data generators, and gradient diagnostics.

## 28. Varying the shortening factor during training

The original Hourglass paper describes *shorten factor dropout*: during training, sample the shortening factor from a predefined set rather than always using the same number. [1]

In our educational variant, the average-pooling model samples between `k=2` and `k=3`. Because average pooling and repeat upsampling do not have factor-specific learned weights, they can share the same Transformer parameters across both choices.

The benchmark evaluates this model at `k=3` after training. It reaches **14.9%** on familiar period-four sequences and **9.0%** under period shift.

Compared with fixed `k=3`, its in-distribution result is slightly lower, while its shifted result is higher. That is consistent with the intuition that varying group boundaries may reduce specialization, although a three-seed, 95-step toy benchmark is insufficient for a general conclusion.

There is a technical caveat for learned linear pooling: a projection from `k*D` inputs to `D` outputs has factor-dependent shape. Our dynamic-factor implementation therefore applies only to average-pooling variants.

## 29. Are the models actually faster?

For the 95-step local CPU benchmark, the measured mean training times were approximately:

- Vanilla four-layer Transformer: **1.195 seconds**.
- Average Hourglass, `k=2`: **0.820 seconds**.
- Average Hourglass, `k=3`: **0.778 seconds**.
- Average Hourglass, `k=4`: **0.793 seconds**.
- Linear Hourglass, `k=3`: **0.981 seconds**.

These results are from a small PyTorch process running on one available CPU environment. They include Python overhead, synthetic data generation, automatic differentiation, and optimizer steps. They are not stable universal throughput claims.

At larger sequence lengths, the attention-score reduction can become more significant. At very short lengths, kernel selection, memory layout, CPU scheduling, and resampling overhead can dominate.

The paper's key efficiency claim is a quality-versus-compute result over realistic modeling workloads, rather than a guarantee that every Hourglass forward pass runs faster on every device. [1]

If evaluating the method for a real system, I would measure wall-clock training speed, peak allocated device memory, step throughput, loss at fixed compute, and end-to-end generation latency separately. The last metric deserves special attention because hierarchical resampling can complicate incremental autoregressive decoding and caching.

## 30. The paper's published language and image results

The original authors evaluated much larger networks than anything in our lab. Several results are worth understanding precisely.

On the byte-level **enwik8** language modeling benchmark, the paper reports **0.98 bits per character** for a 146-million-parameter Hourglass configuration. Its comparison includes a Transformer-XL result of 0.99 bits per character with 277 million parameters. The evaluation conditions and model designs differ, so this should be read as evidence of competitive efficiency, not as a perfect controlled match. [1, 2]

For one of the more controlled validation comparisons, a six-layer full-resolution Transformer-XL baseline obtained **1.182 BPC**, while an Hourglass configuration with two outer layers, four shortened middle layers, and two final full-resolution layers obtained **1.143 BPC**. The latter had more parameters but processed much of its network at reduced resolution. [1]

On **ImageNet32 autoregressive image generation**, the paper reports **3.741 bits per dimension**, compared with 3.758 for a preceding autoregressive Transformer result. That was an important result for its publication context. It is not a statement about today's overall image-generation leaderboard. [1]

For **ImageNet64**, the paper reports **3.443 bits per dimension** using efficient attention on its full-resolution layers. This illustrates that shortening and alternative attention mechanisms can be combined, rather than treated as mutually exclusive approaches. [1, 5]

These results support the broader idea that some network capacity can be moved to shorter sequence resolutions without discarding all token-level modeling ability.

## 31. What the paper's other ablations reveal

Beyond the headline results, the original paper offers unusually practical architectural evidence.

First, the authors study how many shortened layers to use. With fixed fine-resolution layers, adding more shortened processing generally improves the tested language and image metrics. That suggests the compressed stages are doing useful modeling work, not merely acting as cheap placeholders. [1]

Second, they compare where to place fine-resolution layers. The paper reports that a balanced distribution of pre- and post-shortening layers works well in their experiments. Keeping exact token distinctions available on both sides of compression is useful for autoregressive tasks.

Third, they compare upsampling methods. Pure repetition is cheap, but learned and attention-based reconstruction can be more expressive. In a next-token model, the ability to distinguish fine positions matters.

Fourth, they examine pooling. The best choice depends on the dataset and surrounding architecture, which argues against treating average pooling as an obviously optimal default.

Fifth, the paper's shorten factor dropout experiment finds benefits from training with multiple factors rather than committing to one grouping. That idea is especially interesting for text, where fixed-size groups can cut through meaningful linguistic boundaries.

Our workshop intentionally isolates only a subset of these choices. We provide working code for average and linear resampling, not the paper's full attention-based resampling machinery or original Transformer-XL relative attention.

## 32. Why a fast small model is not a faithful reproduction

A research tutorial has an obligation to state the differences between the original work and the educational model.

Our network uses width 32, a vocabulary of 32 synthetic symbols, four attention heads, and a short repeat task. The published Hourglass models use much larger networks, real byte or image datasets, more sophisticated attention choices, and far longer training schedules.

Our implementation also uses simple sinusoidal positions. The paper's Transformer-XL-based setups use relative positional attention. We use current PyTorch scaled dot-product attention rather than recreating the authors' full JAX/Trax training code. [1, 2, 4]

The 95-step training budget is explicitly chosen to reveal early optimization behavior on CPU, not to establish converged perplexity. Model sizes are matched for the vanilla and average-pooling variants but not for learned linear pooling. No claimed speedup includes a careful GPU kernel analysis.

What the lab **does** reproduce directly are the architectural ideas: hierarchical reduction, reconstruction, skip connections, the correct autoregressive shift, and the fact that weakening those safeguards can change model behavior.

We also evaluate those ideas with an independent generator, predefined seeds, saved metrics, and executable assertions.

The most important result is arguably the causal test. A model that violates the temporal contract is not an alternative implementation of the same objective.

## 33. Practical failure modes and engineering safeguards

If you are implementing hierarchical autoregressive models for real workloads, here is what I would verify before believing the first attractive loss curve.

**First, check every non-attention operation for leakage.** Pooling, convolution, patch construction, downsampling, cross-resolution attention, and interpolation can mix temporal positions outside the mask you carefully constructed for attention.

**Second, test dependency invariants numerically.** For every output index, modify future input tokens and ensure the output remains unchanged. Use random inputs, several lengths, edge cases, nested hierarchies, and both training and evaluation modes where appropriate.

**Third, evaluate at fine and coarse boundaries.** Error rates may cluster around token-group boundaries. A strong mean score can hide a systematic weakness on the first or last token of each group.

**Fourth, measure actual resources.** Report loss or perplexity at fixed optimizer updates, at fixed wall-clock time, and where feasible at fixed FLOPs. Those are not interchangeable.

**Fifth, control data leakage independently from model causality.** A perfectly causal architecture can still benefit from train/test contamination or test-set hyperparameter tuning.

**Sixth, probe distribution shifts.** Change period, sequence length, local density, rare-token placement, and the alignment of informative features with pooling boundaries.

**Seventh, test inference, not only teacher-forced training.** An implementation that makes fast parallel training predictions may still need expensive recomputation during autoregressive generation. Caching and group-boundary update behavior matter.

**Eighth, retain simple baselines.** Compare against a standard Transformer, a smaller standard Transformer at similar compute, and a model that processes fewer tokens without a learned hierarchy. Otherwise, it is difficult to know what actually produced the gain.

## 34. Extending the lab to real data

Once the synthetic benchmark is working, the next natural step is a small real-language experiment.

One option is **enwik8**, the same byte-level benchmark used in the paper. You can adapt the training generator to sample contiguous byte windows from the training partition, then train on `window[:-1]` and evaluate against `window[1:]`.

A rigorous evaluation should use separate training, validation, and test partitions, report bits per character, and prohibit tuning on the held-out test set. Be careful not to interpret overlapping evaluation windows as independent samples when estimating uncertainty.

The paper uses overlapping windows for test evaluation while scoring only the final part of each window, so that previous bytes provide context without being counted repeatedly in the reported loss. [1]

A second option is **Tiny Shakespeare**, primarily for functional correctness and iteration speed. It is small enough that overfitting and memorization are meaningful concerns. Use it as a software exercise, not as evidence of competitiveness with large language models.

A third option is long-form **code completion** on permitted open-source data. Code contains repeated names, local syntax, and longer-range dependencies, which can create a rich tension between short and coarse information. To avoid contaminated evaluation, split by project rather than randomly mixing neighboring lines from the same repository into training and test.

Our supplied repository runs offline and does not automatically fetch these corpora. A real-data extension should make the download source, license, preprocessing, splits, and exact dataset version explicit.

## 35. Open research directions

The Hourglass idea has several extensions that remain interesting even as attention kernels improve.

**Adaptive boundaries.** Fixed windows may merge semantically unrelated tokens while splitting useful chunks. Could a learned boundary predictor shorten according to information content without creating an invalid future-looking decision?

**Content-aware causal resampling.** Can cross-resolution attention preserve difficult local details while keeping a formally verified dependency graph?

**Hierarchical cache design.** If the model is used one token at a time, can the coarse representations be updated only when a new group becomes complete, avoiding redundant computation?

**Variable-rate hierarchies.** Can a model allocate more resolution to code syntax, rare entity mentions, or rapidly changing signals, while treating low-information regions more coarsely?

**Hybrid efficient attention.** Shortening changes the length at which attention operates. Sparse, windowed, recurrent, and low-rank mechanisms change the attention operation itself. The two ideas can be combined, and the original paper provides an early example by evaluating Reformer-style attention. [1, 5]

**Mechanistic interpretability across scales.** What kinds of features are represented at full resolution versus shortened resolution? Does the coarse stage preserve semantics but discard exact surface form? Can we trace a prediction back through cross-resolution paths?

**Out-of-distribution robustness.** Our toy experiment already found a sharp failure when a stable recurrence changed. A larger study could separate the effect of resampling choice, positional encoding, dataset diversity, and training budget.

Related research such as Funnel-Transformer and MEGABYTE shows that multiscale sequence processing is a broader architectural direction, although those systems solve different tasks and use different data flows. [7, 8]

## 36. Reproducibility: run the complete project

The companion repository is called `hourglass_transformer_lab`. It is deliberately independent of labml.ai and the authors' official Trax code. All results labeled "our experiment" originate from its synthetic benchmark.

Install requirements:

```bash
cd hourglass_transformer_lab
python -m pip install -r requirements.txt
```

Run all 25 correctness tests:

```bash
python -m unittest discover -s tests -v
```

See a minimal causal versus non-causal pooling example:

```bash
python -m hourglass_lab.demo
```

Reproduce the saved three-seed experiment:

```bash
python -m hourglass_lab.benchmark \
  --seeds 3 \
  --steps 95 \
  --batch-size 12 \
  --eval-size 128 \
  --output results/benchmark.json
```

Generate the included figures from benchmark JSON:

```bash
python plot_results.py
```

Try a longer study, knowing that results will differ from the supplied short-budget benchmark:

```bash
python -m hourglass_lab.benchmark \
  --seeds 5 \
  --steps 500 \
  --batch-size 24 \
  --output results/longer_study.json
```

The source also supports restricting a run to specified configurations with `--only`, which is useful for debugging or smaller computers.

All results are stochastic. CPU speed can vary dramatically between environments. The exact reported numbers above are from three seeds on one local CPU session with PyTorch 2.10.0. They should be treated as a reproducibility target to investigate, not a promised result on every machine.

The files `BENCHMARK_NOTES.md` and `README.md` describe the experimental controls and the boundary between our lab and the original paper. The folder `figures/` contains the charts and the causal-shortening diagram used in this tutorial.

---

## The conclusion

The Hourglass Transformer is easy to summarize as a U-Net for language. That analogy is useful, but it leaves out what makes the paper technically distinctive.

The authors found a way to use a hierarchy of sequence resolutions while preserving autoregressive language modeling. The shortened attention layers reduce the expensive part of computation. The full-resolution layers and skip connections retain detail. And the additional right shift protects the probability model from future information that pooling would otherwise expose.

Our independent experiments add a caution. Under a tiny, fixed training budget, learned linear resampling can make a striking difference on structured sequences. Yet a change in the structure of the data can make that advantage disappear. A strong benchmark must measure both the quality of the predictions and whether the predictions were produced legally.

The takeaway is broader than Hourglass:

**A more efficient model is not simply one that computes fewer operations. It is one that preserves the right information, at the right resolution, under the right causal constraints, and does so for less measured cost at a comparable level of quality.**

That is the standard worth applying to every new long-context architecture.

---

## References and further reading

**[1]** Piotr Nawrot, Szymon Tworkowski, Michał Tyrolski, Łukasz Kaiser, Yuhuai Wu, Christian Szegedy, and Henryk Michalewski. *Hierarchical Transformers Are More Efficient Language Models*. Findings of NAACL 2022. [Paper and official publication](https://aclanthology.org/2022.findings-naacl.117/). [Full HTML paper](https://arxiv.org/html/2110.13711). [arXiv record](https://arxiv.org/abs/2110.13711).

**[2]** Zihang Dai et al. *Transformer-XL: Attentive Language Models beyond a Fixed-Length Context*. ACL 2019. [Paper](https://aclanthology.org/P19-1285/).

**[3]** labml.ai. *Annotated PyTorch Hourglass implementation*, including shift-right, average pooling, naive upsampling, and unimplemented alternative resampling stubs. [Annotated code](https://nn.labml.ai/transformers/hour_glass/index.html). [Source on GitHub](https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/hour_glass/__init__.py).

**[4]** Google Trax. *Official Hourglass source implementation*. [Source on GitHub](https://github.com/google/trax/blob/master/trax/models/research/hourglass.py).

**[5]** Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. *Reformer: The Efficient Transformer*. ICLR 2020. [arXiv](https://arxiv.org/abs/2001.04451).

**[6]** Olaf Ronneberger, Philipp Fischer, and Thomas Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. 2015. [arXiv](https://arxiv.org/abs/1505.04597).

**[7]** Zihang Dai et al. *Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing*. NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/2cd2915e69546904e4e5d4a2ac9e1652-Abstract.html).

**[8]** Lili Yu et al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers*. NeurIPS 2023. [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f8f78f8043f35890181a824e53a57134-Abstract-Conference.html).

**[9]** PyTorch. *Scaled Dot Product Attention*, implementation and backend behavior. [Official API documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

**[10]** PyTorch. *Average Pooling in One Dimension*. [Official API documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.avg_pool1d.html).

*The companion code is an independent educational implementation. It is not affiliated with or endorsed by the original researchers, labml.ai, Google Research, or OpenAI.*
