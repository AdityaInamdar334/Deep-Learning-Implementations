# A Single Neuron Made Qwen Talk About Tomatoes. Here Is What That Actually Tells Us About Language Models.

## A practical deep dive into activation steering, the residual stream, SwiGLU neurons, causal interventions, model damage, and the limits of mechanistic interpretability

A researcher opens a four-billion-parameter language model, chooses one position inside a feed-forward layer, and pushes that activation upward.

The model begins talking about tomatoes.

Another intervention makes it speak like a fictional villain. A third prompts it to invent an extraterrestrial identity. Nothing was fine-tuned. No new training examples were used to update the network. The intervention takes place while the model is running.

These demonstrations are genuinely fascinating. They also invite an explanation that is almost certainly too simple: perhaps the network has one neuron for tomatoes, another for villains, and a third for aliens. Find the right neurons, turn them up, and read the model's mind.

That is not what the evidence establishes.

A language model is a nonlinear dynamical computation, not a filing cabinet of named concepts. One neuron can influence a behavior without being the sole place that behavior is represented. An intervention can be causally effective without being a faithful explanation. A direction discovered by contrasting two sets of prompts can work because of the words in those prompts rather than the abstract property we hoped it would capture.

The useful question is therefore more demanding:

**How do we turn an exciting steering demonstration into an experiment that survives controls, distribution shift, and a careful measurement of what else broke?**

This tutorial builds that experiment from scratch. We will unpack residual streams and gated MLPs, derive steering vectors from activations, implement reversible PyTorch hooks, intervene on individual intermediate units, inspect a limited weight-projection edit, and measure target behavior against off-target effects. We will then examine the open-source Neuron Explorer project and discuss why automated descriptions of neurons need independent testing.

The companion repository includes a CPU-only PyTorch lab with a four-block residual network, five-seed benchmarks, eleven correctness tests, a one-neuron intervention, a matched random-direction control, an off-target concept control, and an optional real Qwen3-0.6B example. **The default benchmark is an educational synthetic network, not an experiment on Qwen3.5-4B.**

The distinction matters throughout this article.

---

## Contents

1. The neuron that changed a model's personality
2. What the original demonstrations actually show
3. Qwen3.5 is not an ordinary Transformer stack
4. The residual stream as a computational workspace
5. Why features are directions, not necessarily neurons
6. Inside the SwiGLU feed-forward network
7. The difference-of-means steering vector
8. Building a contrastive dataset without fooling ourselves
9. Capturing hidden states correctly
10. Adding a steering vector with a reversible hook
11. The difference between steering and fine-tuning
12. A controlled PyTorch laboratory
13. What our five-seed benchmark actually measured
14. The dose-response curve and model damage
15. Why random directions and off-target controls matter
16. Intervening on one gated MLP unit
17. Why one-nueron effects do not establish monosemanticity
18. Projection edits and the danger of saying a feature is deleted
19. Where sparse autoencoders fit into the story
20. Inside the original Neuron Explorer repository
21. Automated interpretability is hypothesis generation
22. A minimal experiment with an actual Qwen checkpoint
23. Building a credible steering benchmark
24. Statistical pitfalls and failure modes
25. Safety, misuse, and the refusal-direction literature
26. What this means for researchers and production systems
27. Reproducing the lab
28. Limitations
29. Final perspective
30. References

---

## 1. The neuron that changed a model's personality

The video motivating this article describes a set of interventions on Qwen models. In one example, a unit associated with tomato-related text is pushed, and model outputs become strikingly tomato-focused. In another, a different intervention appears to amplify villain-related language. The speaker also examines whether an abstract property such as sadness can be controlled by a single MLP neuron.

There are two families of experiments mixed together here.

The first changes a **direction in the residual stream**. A direction is a vector with as many coordinates as the hidden width of the model. It is generally not aligned with a particular neuron. The model continues its forward pass after we add that vector.

The second changes an **individual intermediate MLP activation**. This selects one coordinate of a feed-forward layer's expansion space. The coordinate's down-projection weight then converts its change into a vector written into the residual stream.

The distinction is not cosmetic. It changes what the intervention means, what is being measured, and how strongly we should interpret the result.

The third technique discussed in the video is **weight editing**. Instead of temporarily changing an activation, a researcher alters learned weight matrices so later computations change even after all hooks have been removed. Some weight-editing work has focused on safety-related refusal behavior. We will discuss that research and its implications, but the accompanying executable lab only edits harmless synthetic mood and topic features.

A useful experimental vocabulary is:

- **Observation:** a unit often activates on tomato-related inputs.
- **Association:** the unit's activity predicts the presence of tomato-related text on held-out examples.
- **Intervention:** forcing the unit to a different value changes the model's output.
- **Specificity:** the intervention affects tomato-related behavior more than unrelated behavior.
- **Mechanism:** the complete computational path explains why and when the intervention works.

These are increasingly strong statements. A viral generation sample usually supports an intervention claim. It rarely establishes the complete mechanism.

The activation-engineering literature provides the foundational tools for making these questions testable. Turner and colleagues introduced systematic inference-time activation additions, while contrastive activation addition later made paired contrast examples central to the approach [1, 2].

## 2. What the original demonstrations actually show

Before touching code, consider the source boundaries.

The supplied video is an educational walkthrough. Its demonstrations include residual steering for emotional tone, targeted MLP interventions for narrower concepts, random-direction controls, and a discussion of permanent weight edits. The speaker reports specific neuron locations and outputs, but those particular claims are video demonstrations, not independently replicated results in this article.

There is, however, an independently inspectable repository behind an important part of the story. Lyra's **Neuron Explorer** is an open-source project for Qwen3.5-4B. Its stated strategy is to perturb every intermediate MLP neuron in a selected layer, save the resulting generations, compare against random controls, and make the results browsable. Its research log describes a 9,216-neuron sweep in layer 22, along with prompt-based rollouts and follow-up investigations [3, 4].

The correct way to read these demonstrations is not that a single, isolated neuron *contains* a concept. It is that individual intermediate coordinates can sometimes serve as powerful intervention points in the computation producing a concept.

There is also a model-size issue. The video moves between Qwen3-0.6B, a conventional small decoder model, and Qwen3.5-4B, a different, hybrid architecture. Conclusions drawn from one do not automatically transfer to the other, even when the names look similar.

## 3. Qwen3.5 is not an ordinary Transformer stack

For the smaller model, the published Qwen3-0.6B checkpoint has 28 decoder layers. Its hidden width is 1,024. This makes it an accessible model for experimenting with standard decoder-layer hooks on a modest machine [5].

For the larger model, the official Qwen3.5-4B model card describes:

- 32 decoder layers.
- A 2,560-dimensional hidden representation.
- A 9,216-dimensional feed-forward intermediate representation.
- A hybrid block layout consisting of three Gated DeltaNet layers followed by one gated full-attention layer, repeated eight times.
- A natively multimodal model family, even though our discussion centers on text-side computations [6].

The attention part is important. Saying that every Qwen3.5 block contains ordinary softmax self-attention would be incorrect. Three quarters of the text blocks use a Gated DeltaNet recurrent mechanism, while the remaining blocks use full attention. The feed-forward side still provides an intermediate representation that can be analyzed and manipulated.

That gives us two different spaces to distinguish:

**The residual space:** 2,560 coordinates per token in Qwen3.5-4B.

**The MLP intermediate space:** 9,216 gated coordinates per token in that same model.

Changing a residual direction means adding a 2,560-dimensional vector. Changing one MLP unit means changing one number among those 9,216 intermediate values. The down-projection converts the second kind of change into the first.

This distinction also explains why a coordinate identifier such as `layer 22, neuron 2742` is model-specific. It is not a universal biological-style neuron address. It depends on the checkpoint, architecture, intermediate feature definition, layer numbering, and even whether counting begins at zero.

## 4. The residual stream as a computational workspace

Consider a simplified residual block:

```python
x = x + attention_or_mixer(norm1(x))
x = x + mlp(norm2(x))
```

The tensor `x` is the residual stream. In a standard autoregressive decoder, its shape is commonly:

```text
[batch_size, sequence_length, hidden_width]
```

Each block receives the current representation and writes an update. That makes the residual stream a natural place for experiments: insert a vector at one layer and observe how later blocks and the output head respond.

It is tempting to call the stream a semantic coordinate system where one axis is happiness and another is fruit. That picture is misleading. The trained network can represent concepts as combinations of coordinates; individual axes generally have no stable human interpretation in the absence of additional constraints.

What matters is a **direction** through the high-dimensional representation space.

Suppose an experimenter finds a direction `d` that appears more frequently when a model processes sad texts than happy texts. At layer `l`, an intervention can be written:

```text
h_steered = h_original + alpha * d
```

Here `alpha` is the intervention strength. Positive and negative strengths traverse opposite directions. The later model blocks then transform the modified representation.

The key subtlety is that the direction may encode several correlated properties at once. If all sad examples are first-person narratives and all happy examples are weather reports, the difference will mix tone, genre, perspective, and vocabulary.

The vector is only as conceptually clean as the experimental design used to discover it.

## 5. Why features are directions, not necessarily neurons

The simplest analogy is a photograph.

A pixel can influence the visibility of a face, but the face is not stored in that pixel. It is represented by a pattern across many pixels. Some neural features work similarly across activation coordinates.

The argument becomes stronger in models exhibiting **superposition**. Research from Anthropic shows how networks can use overlapping directions to represent more useful features than the number of available dimensions. Individual units can respond to multiple unrelated properties because feature vectors share representation space [7].

A single activation coordinate can therefore have multiple meanings depending on context. Conversely, a broad behavioral feature can depend on a coordinated set of units across layers.

This is why the video's failed search for a unique sadness neuron is illuminating. A broad property such as narrative sadness probably requires many representational components: emotional vocabulary, setting, temporality, character goals, negation, loss, genre, and learned writing conventions. A narrower concept such as a particular food word may offer a more detectable and manipulable localized handle.

But even a sharply selective tomato-related neuron could be one of several redundant or overlapping routes to that concept.

To claim monosemanticity, we would want much more than one memorable output. We would test activations across diverse natural text, false positives, semantically adjacent terms, negated examples, multilingual instances, unusual contexts, and independently generated prompts.

## 6. Inside the SwiGLU feed-forward network

Modern decoder models often use a gated MLP rather than the simple `Linear -> ReLU -> Linear` picture. In the Qwen family, the feed-forward computation takes the form:

```python
hidden = silu(gate_proj(x)) * up_proj(x)
mlp_output = down_proj(hidden)
```

A PyTorch implementation looks like this:

```python
class SwiGLU(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate = torch.nn.Linear(hidden_size, intermediate_size)
        self.up = torch.nn.Linear(hidden_size, intermediate_size)
        self.down = torch.nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        activated = torch.nn.functional.silu(self.gate(x)) * self.up(x)
        return self.down(activated)
```

The actual Qwen3.5 checkpoint uses its own model implementation and may omit linear biases. The small class above illustrates the same gated structure rather than duplicating every checkpoint detail.

The intermediate vector contains the individual values that neuron-steering projects often manipulate. A neuron in this setting is **not** just one row of `up_proj`, because the gate also matters. It is the resulting gated intermediate coordinate, formed by the product of two learned transformations.

Now consider the down-projection matrix. For a `hidden_size` of 2,560 and an `intermediate_size` of 9,216, a framework such as PyTorch commonly stores `down_proj.weight` with shape:

```text
[2560, 9216]
```

Column `j` describes how intermediate unit `j` contributes to the residual-width output. If the unit's activation changes by `delta`, its direct MLP output contribution changes by:

```text
delta * down_proj.weight[:, j]
```

This is an exact local identity for a linear down-projection with all other intermediate coordinates held fixed. What happens after that is more complicated because downstream layers, normalization, and output logits respond to the modified residual vector.

This explains the power of neuron interventions: one intermediate coordinate has a learned vector of potential consequences.

It also explains their limits: a neuron may have a strong local output direction without being the unique source of a high-level behavior.

## 7. The difference-of-means steering vector

The simplest practical feature discovery method is the **difference of means**.

Collect two sets of examples: one expressing the target property and one expressing its contrast. Pass both through the model with ordinary forward passes and capture activations at a chosen layer.

For each layer, compute:

```python
positive_mean = positive_states.mean(dim=0)
negative_mean = negative_states.mean(dim=0)

direction = positive_mean - negative_mean
direction = torch.nn.functional.normalize(direction, dim=0)
```

If we contrast gloomy and joyful examples, the result is an estimated gloomy direction. But estimated is the important word. The vector is not uniquely determined by the property itself. It depends on the examples, prompt template, selected token positions, architecture, and layer.

Two closely related techniques appear in the research literature.

**Activation Addition**, or ActAdd, extracts differences from contrast examples and adds them during inference [1].

**Contrastive Activation Addition**, or CAA, uses paired examples to estimate behavioral directions, including differences between correct and incorrect responses or different dispositions [2].

Their significance is not just that steering works. They provide a repeatable experimental protocol in which an activation intervention can be isolated from ordinary prompt instructions.

That is the core idea we will reproduce.

## 8. Building a contrastive dataset without fooling ourselves

The naive version looks like this:

```text
Positive: "I feel joyful today."
Negative: "I feel gloomy today."
```

It is useful as a demonstration. It is not sufficient evidence of an abstract emotion representation.

The dataset contains a strong lexical shortcut. The model could distinguish the two sets by detecting `joyful` versus `gloomy`, without using any representation resembling an emotional state.

A stronger contrast experiment should deliberately vary the surface form while controlling the target factor:

```text
Positive: "The traveler celebrated after reaching home."
Negative: "The traveler mourned after reaching home."

Positive: "Her friends gathered for a joyful reunion."
Negative: "Her friends gathered for a painful farewell."
```

Even here, we need to worry about lexical overlap, structure, and the distribution of sentence length. Matching contexts helps but does not eliminate the issue.

For proper evaluation, use at least three non-overlapping groups:

1. A discovery split to estimate the direction.
2. A validation split to select the layer and intervention strength.
3. A test split to measure the final result once the choices are frozen.

The test split should include new templates, different lexical realizations, and examples that combine target and non-target properties in unusual ways.

The companion experiment intentionally uses simple artificial tokens to make the problem observable. It also holds out a different set of nuisance tokens at evaluation time. That does not solve all possible confounds, but it makes the experiment more reproducible than inspecting a few hand-picked completions.

## 9. Capturing hidden states correctly

For a causal language model loaded through Hugging Face Transformers, one straightforward option is:

```python
with torch.no_grad():
    outputs = model(
        **tokenized,
        output_hidden_states=True,
        use_cache=False,
    )

hidden_states = outputs.hidden_states
```

Be careful with indexing. In many decoder implementations, `hidden_states[0]` refers to the input embedding representation, while `hidden_states[layer_index + 1]` is the hidden state returned after decoder block `layer_index`.

If the model has 28 layers, the tuple can contain 29 elements. That is not a 29-layer model. It is the embedding state plus 28 block states.

Token selection matters as much as layer selection. Averaging all token positions, choosing the final prompt token, capturing answer-token positions, and pooling over a document are four different measurements.

For classification-like prompts, a final-token representation may be appropriate. For a feature that appears only in one word, token-aligned measurements may be more interpretable. If you average all tokens, most of the mean can reflect sentence structure and common filler rather than the target word.

Padding also matters. If your batched inputs have different lengths, do not accidentally capture the final padding token instead of the final real token. Use the attention mask to locate the intended positions.

In the optional Qwen3-0.6B experiment, this is how we identify the last real token:

```python
positions = batch["attention_mask"].sum(-1) - 1
rows = torch.arange(len(texts), device=device)
activations = states[layer_index + 1][rows, positions]
```

This example assumes right-padded inputs. Different padding arrangements require different indexing logic.

## 10. Adding a steering vector with a reversible hook

PyTorch forward hooks let us intercept the output of a module without editing its stored parameters.

The useful abstraction is a context manager that guarantees cleanup:

```python
from contextlib import contextmanager

@contextmanager
def steer_residual(model, layer_id, direction, strength):
    def hook(module, inputs, output):
        return output + strength * direction.to(
            output.device, output.dtype
        )

    handle = model.layers[layer_id].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
```

The temporary intervention now reads naturally:

```python
baseline = model(tokens)

with steer_residual(model, layer_id=2, direction=d, strength=2.0):
    steered = model(tokens)

restored = model(tokens)
```

The first and third computations should match in a deterministic inference setting. If they do not, the hook may not have been removed, the model may be in training mode, or some other state has changed.

For an actual Hugging Face decoder block, the output may be a tuple rather than a single tensor. Your hook must preserve that structure, modifying only the hidden-state component. The included optional Qwen example handles both tensor and tuple return values.

A final engineering warning: autoregressive generation frequently uses a KV cache or recurrent state. If an intervention is applied inconsistently across prefill and decoding, the comparison can mean something different from the experiment you intended. Reusing a cache created before steering may preserve unsteered information. Specify your cache policy and check it.

## 11. The difference between steering and fine-tuning

These interventions change different parts of the system.

**Prompting** changes the input sequence and relies on the model interpreting that instruction.

**Activation steering** changes an internal representation while the model runs. The parameters remain untouched, and the change can be undone simply by removing the hook.

**Fine-tuning** changes learned parameters through optimization on training data. The modification persists and may affect a wide range of contexts.

**Weight projection editing** also changes stored parameters, but it does so by a chosen algebraic transformation rather than a training loop.

Why does this distinction matter? Because a successful activation intervention does not prove that the same effect can be made reliable through a permanent weight edit. The later computation may depend on how and where the intervention is injected.

Likewise, a permanent edit that lowers a measured feature coordinate does not guarantee that all behavior associated with that concept has been removed. Other pathways may regenerate the feature or produce the same behavior without it.

## 12. A controlled PyTorch laboratory

Instead of downloading several gigabytes of weights and immediately chasing a visually convincing completion, let us begin with an experiment where we know exactly what the network is supposed to represent.

The included lab trains a small, four-block residual model with SwiGLU MLPs and two classifier heads. Its short inputs contain two independent artificial attributes:

- **Mood:** joyful or gloomy.
- **Topic:** tomato or stone.

Additional filler tokens are intentionally irrelevant. Different filler vocabularies appear in training and testing, and the token order is randomized.

The network must predict both attributes. We then learn a mood direction from one set of examples and intervene on a disjoint evaluation set. Because the task has two labels, we can track whether changing mood also damages topic prediction.

The model core is:

```python
class ResidualBlock(torch.nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, dim * 2)

    def forward(self, x):
        return x + self.mlp(self.norm(x))
```

The classifier embeds each token, averages the token embeddings, passes the result through four residual blocks, normalizes the representation, and produces independent mood and topic logits.

This is deliberately **not a language model**. It does not autoregressively generate text, does not use self-attention or Gated DeltaNet, and does not learn semantics from a web-scale corpus. We are reproducing the intervention algebra in an inspectable setting, not reproducing the original Qwen demonstrations.

We train with a conventional supervised objective:

```python
mood_logits, topic_logits = model(tokens)
loss = (
    torch.nn.functional.cross_entropy(mood_logits, mood_labels)
    + torch.nn.functional.cross_entropy(topic_logits, topic_labels)
)
loss.backward()
optimizer.step()
```

Once the model is trained, all steering experiments operate in evaluation mode, without gradient updates.

## 13. What our five-seed benchmark actually measured

We ran five deterministic training seeds, each with 160 training steps and 512 held-out examples. The steering direction was estimated from a separate synthetic discovery set. The selected layer was fixed at the third residual block, indexed as layer 2.

Here are the mean results from that exact setup.

**Unmodified network:** 91.1% mood classification accuracy, 81.5% topic accuracy, and a mean 46.8% probability assigned to the gloomy class.

**Mood direction, strength +0.5:** gloomy probability rose to 50.1%. Mood accuracy was 92.9%, with a mood KL divergence of 0.027 relative to baseline.

**Mood direction, strength +1:** gloomy probability rose to 53.7%. Mood accuracy was 91.8%, with mood KL of 0.110.

**Mood direction, strength +2:** gloomy probability rose to 60.7%. Mood accuracy was 90.0%, with mood KL of 0.425.

**Mood direction, strength +4:** gloomy probability reached 72.5%, but mood accuracy fell to 78.7%, and mood KL increased to 1.378.

**Equal-strength random direction, strength +2:** gloomy probability was 48.5%, with mood KL of 0.033.

**Off-target topic direction, strength +2:** gloomy probability was 47.0%, with mood KL of 0.022.

**Negative mood direction, strength -2:** gloomy probability declined to 34.9%.

These numbers support a specific, modest conclusion: **a direction estimated from contrastive activations can causally shift one output distribution in a trained network more strongly than a matched random direction.** They do not show that a real language model became sad, that a behavior was fully localized, or that the intervention would generalize to new reasoning tasks.

The baseline accuracy is not perfect partly because held-out examples use nuisance tokens outside the training nuisance vocabulary. The intervention was evaluated against this specific distribution.

A more important warning is that mean target-class probability does not equal better task performance. The labels in our evaluation remain unchanged after steering. Pushing every example toward gloomy will help some examples and hurt others. That is why classification accuracy decreases at the strongest dose even as gloomy probability rises.

This is exactly the sort of distinction a demo of generated text can hide.

## 14. The dose-response curve and model damage

A compelling steering example usually has an effective strength range. Too little intervention produces a negligible effect. Too much can destroy useful information or produce repetitive, incoherent output.

We need to measure both sides of this relationship.

One simple distributional measure is KL divergence. Let `p` be the original output probability distribution over the vocabulary and `q` be the distribution after an intervention. The KL divergence from `p` to `q` measures how much the intervention changes the predicted distribution relative to the original model.

In PyTorch:

```python
def categorical_kl(reference_logits, changed_logits):
    p = torch.nn.functional.softmax(reference_logits, dim=-1)
    log_p = torch.nn.functional.log_softmax(reference_logits, dim=-1)
    log_q = torch.nn.functional.log_softmax(changed_logits, dim=-1)
    return (p * (log_p - log_q)).sum(-1).mean()
```

The direction of KL matters. It is asymmetric. Here we compute the KL of the original distribution relative to the changed one.

In the synthetic experiment, mood KL increased from 0.027 at strength 0.5 to 1.378 at strength 4. The steering intervention was no longer a subtle perturbation of the original behavior.

But **KL is not itself a damage metric in the moral or functional sense**. A good steering intervention is supposed to change the output distribution. The real question is whether it changes what you intended while preserving what you did not intend to change.

That is why we measure task accuracy on unchanged labels, the second topic head, and negative controls. On an actual language model, also evaluate factual recall, reasoning accuracy, instruction following, toxicity, refusal calibration, language consistency, and generation quality on independent prompts.

The correct target is not simply minimum KL. It is a favorable target-behavior change subject to an acceptable off-target performance budget.

## 15. Why random directions and off-target controls matter

A large hidden-state perturbation can alter a model even if it carries no meaningful information. Without controls, it is easy to mistake an ordinary distribution shift for a discovered feature.

We use two primary controls.

**An equal-norm random direction.** We draw a random vector, remove its projection onto the learned mood direction, and normalize it. We then apply the same strength. This tests whether our measured effect is larger than a generic intervention of similar norm.

**An off-target concept direction.** We compute a tomato-versus-stone contrast vector using the same approach and ask whether it changes gloomy probability. This tests whether the method is indiscriminately disrupting the network.

The mood direction at strength +2 raised gloomy probability from 46.8% to 60.7%. The equal-strength random direction reached only 48.5%, and the topic direction reached 47.0%.

This is encouraging, but there is another trap: one random vector is an inadequate statistical reference for a large neuron search.

When screening thousands of units, compare against an empirical null distribution of many random directions with matched norms, layers, prompts, and dose selection. Record not only the most impressive neuron but the full distribution of observed effects.

If you screen 9,216 neurons and select the best-looking one, there is a substantial multiple-comparisons problem. Extreme results are expected even when many candidates are uninteresting.

A rigorous follow-up reserves a new prompt set and re-tests the selected unit after it has been chosen. If the effect disappears, the original selection may have been a search artifact.

## 16. Intervening on one gated MLP unit

A residual-direction experiment changes an entire hidden-width vector. A neuron intervention changes just one coordinate of the gated MLP intermediate tensor.

To do that correctly, we need a hook **between the gated elementwise product and the down-projection**.

In the companion model, we expose this location through a no-op `Identity` module:

```python
class SwiGLU(torch.nn.Module):
    def __init__(self, dim=48, inner=96):
        super().__init__()
        self.gate = torch.nn.Linear(dim, inner)
        self.up = torch.nn.Linear(dim, inner)
        self.tap = torch.nn.Identity()
        self.down = torch.nn.Linear(inner, dim)

    def forward(self, x):
        h = self.tap(
            torch.nn.functional.silu(self.gate(x)) * self.up(x)
        )
        return self.down(h)
```

Now a hook can copy the intermediate tensor and add a chosen amount to one coordinate:

```python
from contextlib import contextmanager

@contextmanager
def steer_neuron(model, layer_id, neuron_id, delta):
    tap = model.layers[layer_id].mlp.tap

    def hook(module, inputs, output):
        result = output.clone()
        result[..., neuron_id] += delta
        return result

    handle = tap.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
```

We select the unit with the largest activation mean difference between the gloomy and joyful discovery sets and move it in the sign suggested by that difference.

In our controlled experiment, this single-unit intervention at an absolute strength of 5 raised gloomy probability to **61.8%**, compared with 46.8% at baseline. Mood accuracy decreased to **88.8%**, and topic accuracy fell to **77.5%**.

This is a causal effect. It is not a claim of one-neuron mood representation.

Notice the specificity loss: topic accuracy also dropped. A neuron can strongly affect a target behavior while having other substantial functions.

## 17. Why one-neuron effects do not establish monosemanticity

Imagine discovering that a neuron fires strongly when the text mentions tomatoes, ketchup, pizza sauce, and red vegetables. This is a plausible candidate for a tomato-related feature. It may even be strikingly useful for steering.

But what does it actually represent?

Several hypotheses fit the same data:

1. The unit selectively recognizes the word tomato.
2. It recognizes a broader class of food-related tokens.
3. It participates in a compositional feature for red ingredients.
4. It indexes the narrative topic currently discussed.
5. It influences model output through an unrelated but correlated direction.

The observation alone does not choose among them.

To investigate, create systematic counterexamples:

```text
"I chopped a tomato for lunch."
"The tomato plant grew in the garden."
"The word tomato appears in this sentence."
"There was no tomato in the recipe."
"The painting depicted a tomato, not a real vegetable."
"A red apple sat next to a pepper."
"The sauce was green and contained no tomato."
```

Then measure the unit's activations across the different contexts. Repeat across languages and tokenizations. Compare to other neurons in the same and neighboring layers. Intervene and check whether expected changes follow.

Crucially, a neuron may be selective but not necessary. If you suppress it, the model might still recognize and generate tomato-related text through other circuits.

That distinction is the difference between **a causal handle** and **a complete representation**.

The latter is a much stronger scientific claim.

## 18. Projection edits and the danger of saying a feature is deleted

The video also discusses permanent weight updates that suppress output along a chosen direction.

The underlying linear algebra is simple. For a unit-norm direction `d` in residual space, the matrix `P = I - d dᵀ` projects a vector onto the subspace orthogonal to `d`.

Rather than allocating the entire projection matrix, we can project a down-projection matrix directly:

```python
with torch.no_grad():
    down.weight.sub_(
        torch.outer(d, d @ down.weight)
    )
```

Our toy model also projects the down-projection **bias**, because a bias vector can contribute to the same direction.

This edit ensures that the edited layer's direct MLP output contribution is orthogonal to `d` to numerical precision. It does **not** prove that the full residual stream no longer contains `d`. The input residual can already contain it, attention or a recurrent mixer can write it, and later MLPs can recreate it.

The companion lab makes this distinction explicit by editing a **copy** of the network, leaving the original weights unchanged for comparison.

In the benchmark, the one-layer projection edit changed mean gloomy probability from 46.8% to **45.6%**, and its mood KL divergence was **0.066**. That limited effect is itself instructive. A correct local projection does not guarantee meaningful global behavior change.

This matters whenever a model-editing technique is described as deleting a concept or suppressing it permanently. The algebra can establish a local property of selected matrices. Only broad empirical testing can establish a reliable behavioral claim.

The refusal-direction literature shows that some safety behaviors are especially sensitive to such edits. Arditi and colleagues demonstrated a low-dimensional refusal-related mechanism across several models [8]. That result is important for auditing safety robustness. The downloadable lab deliberately confines its weight edits to benign synthetic features rather than implementing refusal removal.

## 19. Where sparse autoencoders fit into the story

A natural question follows: if features are often distributed, why study raw neurons at all?

Because raw-neuron studies are cheap, causal interventions are direct, and the intermediate dimensions are already accessible. They provide useful hypotheses about what the network is doing.

However, raw coordinates are not necessarily the best units for interpretation.

Sparse autoencoders attempt to learn a more interpretable dictionary of features from activation data. Conceptually, they map hidden representations to a larger set of mostly inactive learned features, then reconstruct the original activations from those features.

A typical training objective combines reconstruction error with a penalty encouraging sparse latent activations. This makes it possible to represent a concept using a learned direction that is not aligned with a raw network coordinate.

Anthropic's *Towards Monosemanticity* showed how sparse dictionary learning could yield thousands of interpretable features from a smaller underlying activation dimension [9]. Its later scaling work extended this line of research to much larger systems [10].

SAEs also have limitations. Their learned features depend on architecture, training data, sparsity choices, feature count, and training dynamics. Reconstruction error can hide important lost information. A feature that is easy to name may not be causally important.

The most compelling workflow therefore combines methods:

- Discover candidate features using correlations or a sparse dictionary.
- Test interventions on those features.
- Evaluate specificity, necessity, and sufficiency on independent prompts.
- Trace how later components transform the intervention.
- Test whether the explanation holds under distribution shift.

This moves interpretability from an attractive label to a falsifiable account of computation.

## 20. Inside the original Neuron Explorer repository

Lyra's open-source [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) takes a valuable, intervention-first approach. Instead of beginning with a list of tokens that strongly activate a neuron, it asks what happens when that neuron is deliberately pushed during generation [3, 4].

The repository documents sweeps across intermediate units, saved token rollouts, multiple intervention doses, random-direction controls, and a browser interface that can show activation views, neuron-specific outcomes, and connections.

The source README documents a Qwen3.5-4B layer-22 sweep with **9,216 MLP units** and **256 random controls**. It also describes experiments on selected layers, and a data collection pipeline for high-activation contexts in a broad web-text sample.

These are design choices worth learning from.

First, the project preserves **all rollouts**, not just the funniest examples. This reduces the temptation to highlight one surprising completion and discard dozens of boring or inconsistent ones.

Second, it uses multiple prompt families. A unit that affects a self-description prompt may not affect code generation, image descriptions, or open-ended text in the same way.

Third, it measures against random controls. If a neuron-generated response is unusual but a random perturbation produces equally unusual responses, the case for a distinctive feature is weaker.

Fourth, it separates different questions: what activates a unit, what the unit writes, what language appears after steering, and what other units it may influence.

Fifth, it includes the beginnings of an agent-assisted research workflow. Automated systems can inspect candidate neurons, formulate labels, propose new prompts, and run tests. The risk is that the agent becomes very good at inventing plausible descriptions for noisy effects. That is why independent validation is essential.

**Important:** The numbers reported by the repository describe its own experiments. We have not independently recreated its full Qwen3.5-4B sweep or verified every neuron-specific behavioral anecdote.

## 21. Automated interpretability is hypothesis generation

The idea of asking a language model to explain neural units predates the latest neuron explorers. OpenAI's 2023 work used GPT-4 to produce and score natural-language explanations for neurons in GPT-2. The team released generated explanations and associated scores while acknowledging their imperfections [11].

Later tooling, including Neuronpedia, expanded the idea to feature-level descriptions and repeatable explanation pipelines [12].

A sensible automated interpretability agent might follow this procedure:

**Step 1:** Retrieve the highest and lowest activation contexts for a candidate unit.

**Step 2:** Read intervention outcomes over several prompts, including positive and negative perturbations.

**Step 3:** Produce three competing hypotheses rather than a single confident label.

**Step 4:** Design counterexamples that distinguish those hypotheses.

**Step 5:** Run new interventions on prompts not used for discovery.

**Step 6:** Report effect sizes, uncertainty, failures, and possible alternative explanations.

A weak agent writes: "Neuron 4948 is the tomato neuron."

A stronger agent writes: "This intermediate unit shows high activation on several tomato-related contexts and increases tomato-associated tokens in some generations when positively perturbed. The effect overlaps with sauces and other foods, and we have not established that the unit is necessary for recognizing tomatoes."

That second description may sound less dramatic. It is considerably more useful to an actual interpretability researcher.

## 22. A minimal experiment with an actual Qwen checkpoint

The default lab is intentionally small. To bridge its code to a real pretrained model, we also provide `optional_qwen_demo.py`.

It loads Qwen3-0.6B using Hugging Face Transformers, extracts final-token layer activations from matched example sentences, takes a contrastive mean difference, and injects the normalized vector into the selected decoder layer during deterministic generation.

Installation is separate from the offline workshop:

```bash
python -m pip install -U transformers accelerate safetensors
python optional_qwen_demo.py --layer 14 --strength 1.0
```

The script prints an unsteered response, a steered response, and a final unsteered response after the hook is removed.

This is **not a validated claim that Qwen3-0.6B will reliably change emotional tone**. The minimal contrast set is deliberately small, and no language-model checkpoint was downloaded or evaluated during construction of this tutorial. The optional example is included as a starting point, not as a benchmark with reported scores.

To turn it into a proper paper-style experiment, you would need a sufficiently varied discovery corpus, an untouched test suite, multiple random control directions, a layer and strength selection split, a robust text-level evaluator, and measurements of unrelated capabilities.

There is one implementation detail to watch carefully. In a Hugging Face implementation, a decoder block may return a tuple. A safe hook modifies only the first hidden-state tensor and returns the other elements unchanged:

```python
def hook(module, inputs, output):
    if isinstance(output, tuple):
        modified = output[0] + strength * direction.to(
            output[0].device, output[0].dtype
        )
        return (modified,) + output[1:]
    return output + strength * direction.to(output.device, output.dtype)
```

Model internals change between library versions. Inspect the actual module path and return structure before adapting the example to a different checkpoint. Qwen3.5 has a different text and multimodal wrapper from Qwen3-0.6B, so do not blindly reuse the same access path.

## 23. Building a credible steering benchmark

A serious benchmark needs more than a high steering success score.

I would divide the evaluation into five independent questions.

**Question one: Did the target behavior change?**

Measure the proportion of held-out outputs expressing the desired property. For mood steering, use an external sentiment evaluator with human-checked calibration examples. For a topic intervention, measure topic relevance with a blinded evaluator or a carefully designed semantic classifier.

**Question two: Did unrelated capabilities survive?**

Use stable factual questions, reasoning tests, code tasks, multilingual prompts, and structured-format compliance. A model that becomes reliably tomato-focused while forgetting how to answer basic questions has not demonstrated a useful, specific control mechanism.

**Question three: Did the intervention outperform a fair null?**

Evaluate random directions matched for norm, layer, token positions, and intervention schedule. A null with one randomly selected vector is not strong enough for thousands of searched candidates.

**Question four: Did the effect generalize?**

Test prompts outside the templates used to discover or choose the vector. Add negations, paraphrases, distractor concepts, domain changes, and, where appropriate, multilingual examples.

**Question five: Is the identified component necessary, sufficient, or merely influential?**

Increasing a feature may be sufficient to amplify a behavior. Removing it may not be necessary for the behavior to disappear. The difference is central to mechanistic claims.

A well-designed evaluation should include the exact model revision, generation configuration, tokenizer, layer convention, token positions, random seeds, sampling temperature, feature normalization, and checkpoint dtype.

For actual model comparisons, report confidence intervals across prompts and, when possible, multiple model seeds or checkpoints. Do not mistake repeated samples from one prompt for independent experimental units.

## 24. Statistical pitfalls and failure modes

Several subtle errors can make steering results look more compelling than they are.

### Selecting a layer on the test set

If you try every layer and dose on your evaluation prompts, then publish the best combination as a held-out result, you have overfit the evaluation. Use a distinct validation split to pick parameters.

### Comparing unnormalized interventions

A vector from one layer may have a different magnitude from a vector in another. Matching the absolute numeric coefficient `alpha` does not necessarily match the effective perturbation. Consider normalizing by the layer's residual norm as well as tracking raw norms.

### Confusing a residual coordinate with an MLP unit

Setting one hidden dimension is not the same as modifying one gated intermediate feature. The source of the vector and the hook position determine what was changed.

### Forgetting that autoregressive generation is path-dependent

A small early intervention can change the next token. That token then changes every subsequent model input. A dramatic final paragraph may result from a small early probability change amplified by autoregressive feedback, not from a direct global semantic switch at every token.

### Using an LLM judge without checking its biases

A judge may interpret a longer or more emotional output as sadder, regardless of the actual intervention. Include labeled evaluation cases, multiple judges where appropriate, human spot checks, and agreement analysis.

### Treating a high activation as a complete explanation

Top-activation examples describe when a unit is active. They do not establish which downstream paths use that activity.

### Ignoring multiple testing

Scanning 9,216 units, multiple prompts, both signs, and several doses produces many opportunities for apparently remarkable results. Use holdout confirmation, randomized control distributions, and transparent selection criteria.

### Ignoring off-target drift

Strong steering can change factual accuracy, formatting, language, vocabulary diversity, and safety behavior. Evaluate these even when the target effect looks appealing.

## 25. Safety, misuse, and the refusal-direction literature

The same tools that help researchers understand internal representations can expose weaknesses in model safeguards.

A particularly relevant body of research found that refusal behavior in a selection of chat models could be influenced by a low-dimensional residual-stream direction. That result demonstrates that some safety-related behavior may depend on surprisingly localized mechanisms [8].

It also provides a warning against relying on one fragile representational feature as an entire security boundary.

There are at least three distinct risk categories.

**Model robustness:** accidental or unauthorized activation interventions may change model behavior in ways normal prompt-based safety testing does not anticipate.

**Evaluation reliability:** an intervention may preserve benchmark scores while changing behavior in narrow operational contexts. This makes independent evaluation important.

**Interpretability overclaiming:** a vivid localized result may suggest that the model has a human-readable internal moral or intentional state. Behavioral association is not evidence of subjective experience or hidden intent.

The accompanying lab therefore limits executable weight edits to a synthetic mood/topic classifier. It does not provide a recipe for weakening refusal safeguards. The broader point is to study and measure the fragility of representational controls, not to treat those controls as a dependable defense mechanism.

For production systems, a safer architecture combines model-side alignment with external authorization checks, restricted tool permissions, monitoring, logging, and human oversight where stakes justify it. Steering should not be used as a substitute for any of those boundaries.

## 26. What this means for researchers and production systems

For **mechanistic interpretability researchers**, intervention-first exploration is a powerful way to generate causal hypotheses. It becomes much stronger when paired with held-out validation, matched controls, and explicit circuit-level follow-up.

For **model developers**, the existence of strong steering directions suggests that behavior can be sensitive to narrow representation shifts. Test stability under activation perturbations if internal intervention is part of the application design.

For **AI evaluation engineers**, the relevant performance metric is not simply the target attribute's prevalence. It is a multidimensional tradeoff among desired effect, off-target capability preservation, latency, compute, reversibility, and safety.

For **people building open-source research tools**, Neuron Explorer illustrates the value of preserving raw observations and making them accessible for further analysis. A browsable catalog of interventions, controls, and counterexamples is more valuable than a collection of screenshots of particularly entertaining generations.

And for anyone trying to reason about how language models represent concepts, this work suggests a better mental model: concepts are computations distributed across layers and representation spaces, with some intermediate coordinates serving as unusually effective control points.

## 27. Reproducing the lab

Download the companion ZIP, extract it, and enter its root folder:

```bash
cd llm_steering_interpretability_lab
python -m pip install -r requirements.txt
```

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Reproduce the full five-seed study:

```bash
python -m steering_lab.benchmark \
  --seeds 5 \
  --steps 160 \
  --eval-size 512 \
  --output results/benchmark.json
```

On some CPU environments, reducing multithreading overhead can help:

```bash
OMP_NUM_THREADS=1 python -m steering_lab.benchmark \
  --seeds 5 --steps 160 --eval-size 512
```

The output reports mean mood accuracy, topic accuracy, gloomy-class probability, and KL divergence for the baseline and interventions. The full JSON also preserves individual seed runs and summary standard deviations.

For the optional charts:

```bash
python -m pip install matplotlib
python plot_results.py
```

The ZIP contains the following core files:

```text
llm_steering_interpretability_lab/
  README.md
  ARTICLE.md
  BENCHMARK_NOTES.md
  requirements.txt
  optional_qwen_demo.py
  plot_results.py
  steering_lab/
    __init__.py
    model.py
    interventions.py
    benchmark.py
  tests/
    test_lab.py
  results/
    benchmark.json
  figures/
    01_steering_probabilities.png
    02_kl_tradeoff.png
```

The benchmark does not require an API key, hosted inference service, pretrained checkpoint, or GPU. The optional Qwen script is not part of the validated default experiment.

## 28. Limitations

A good experiment makes its scope visible.

First, the default model is a tiny, supervised synthetic classifier. It does not reproduce natural-language generation, learned semantic abstractions, or real Qwen activations. Its results are best interpreted as a demonstration of *causal intervention machinery*.

Second, its representation is encouraged to separate two labels by a supervised objective. A pretrained language model may have much more entangled internal structure. We should not expect the same level of control or specificity.

Third, a mean-difference vector is not automatically causally fundamental. It can capture spurious properties of the discovery set. Our random and off-target controls reduce some confounds but do not eliminate all possible explanations.

Fourth, the reported KL values are measured on two small classifier heads, not a 150,000-plus-token language-model vocabulary. The scale and interpretation of KL differ from an actual LLM measurement.

Fifth, the neuron experiment selects the largest activation contrast from the discovery corpus, then perturbs that unit. It is not a systematic search across multiple checkpoints, dose schedules, and natural language contexts.

Sixth, the projection edit only removes the selected direction from one MLP's direct output contribution. It does not delete an entire concept from the model.

Seventh, the optional real-model script was not executed as part of this offline lab. It is offered as an extension to run on a local checkpoint with compatible libraries and hardware.

Finally, the claims about particular tomato, villain, or alien neurons come from the source video and the original open-source project's own demonstrations. We treat them as motivating case studies, not results independently verified here.

## 29. Final perspective

A neuron can be an excellent place to intervene without being a complete place where a concept lives.

That is perhaps the most important lesson from these demonstrations.

The residual stream provides a shared space in which many directions can carry useful information. Gated MLP units selectively transform that information and write vectors back. Contrastive examples can reveal directions associated with behaviors. Hooks let us perturb them reversibly. Projection edits let us alter selected contributions more permanently. Automated exploration can generate compelling hypotheses about what all of these components are doing.

But each step brings new uncertainty. Correlation does not imply a uniquely represented concept. A causal effect does not imply exclusivity. A projection does not erase all downstream behavior. And an entertaining generation does not substitute for a benchmark.

The research frontier is not simply finding the next tomato neuron. It is learning how to tell when a beautiful interpretability story is true.

That demands a discipline familiar from every other experimental science: controlled interventions, alternative hypotheses, independent measurements, and a willingness to report when a promising explanation fails.

---

## 30. References and further reading

**[1]** Turner, A. M., et al. *Steering Language Models With Activation Engineering*. arXiv:2308.10248. https://arxiv.org/abs/2308.10248

**[2]** Panickssery, N., et al. *Steering Llama 2 via Contrastive Activation Addition*. arXiv:2312.06681. https://arxiv.org/abs/2312.06681

**[3]** Lyra. *neuron-explorer: Steering-first neuron explorer for Qwen3.5-4B*. Source code and documentation. https://github.com/lyramakesmusic/neuron-explorer

**[4]** Lyra. *Neuron Explorer Research Log*. Original experiment design and observations. https://github.com/lyramakesmusic/neuron-explorer/blob/main/research_log.md

**[5]** Qwen Team. *Qwen3-0.6B Model Card*. https://huggingface.co/Qwen/Qwen3-0.6B

**[6]** Qwen Team. *Qwen3.5-4B-Base Model Card*. Architecture and dimensions. https://huggingface.co/Qwen/Qwen3.5-4B-Base

**[7]** Elhage, N., et al. *Toy Models of Superposition*. Anthropic, 2022. https://www.anthropic.com/research/toy-models-of-superposition

**[8]** Arditi, A., et al. *Refusal in Language Models Is Mediated by a Single Direction*. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html

**[9]** Bricken, T., et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic, 2023. https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning

**[10]** Anthropic. *The Engineering Challenges of Scaling Interpretability*. 2024. https://www.anthropic.com/research/engineering-challenges-interpretability

**[11]** OpenAI. *Language Models Can Explain Neurons in Language Models*. 2023. https://openai.com/index/language-models-can-explain-neurons-in-language-models/

**[12]** Neuronpedia. *Auto-Interp Documentation*. https://docs.neuronpedia.org/sparse-autoencoders/auto-interp

**[13]** Hugging Face Transformers. *Qwen3.5 Model Documentation*. https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md

**[14]** The source video transcript supplied for this article, discussing residual steering, gated MLP units, difference of means, model editing, and Neuron Explorer. Examples from the video are distinguished from the companion lab's independently measured results.

---

*Companion code: the downloadable Python workshop associated with this article. Reproduce the benchmark, inspect the hooks, and test the difference between a convincing causal effect and an overstated mechanistic explanation.*
