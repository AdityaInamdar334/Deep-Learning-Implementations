# An AI Edited 0.016% of Qwen's Weights. Its Math Score Jumped From 7.9% to 48.1%.

## Inside MIT CSAIL's Weight-Space Programming experiments, where a language model reads neural networks like programs, edits their weights without gradients, and reveals the difference between repairing a capability and learning one

A neural network with almost a billion parameters gets a mathematics question. It begins reasoning. Instead of reaching an answer, it starts repeating itself, expanding an explanation that never ends. You increase the generation budget. The model keeps writing. Eventually it hits the limit and stops, not because it solved the problem, but because it ran out of room to continue.

Now imagine another language model inspecting the first model's parameters, identifying a narrow mechanism responsible for the repeated reasoning, and directly changing a few columns of two weight matrices. No backpropagation. No fine-tuning dataset. No optimizer state. The model is evaluated again.

In a November 2, 2026 report, researchers from MIT CSAIL say this intervention raised Qwen3.5-0.8B's MATH-500 accuracy from 7.9% to 48.1%. GSM8K rose from 13.8% to 67.1%. The edit addressed 131,072 weight locations, which the authors describe as approximately 0.016% of the model. Their system used GPT-6 Astra to propose and execute the change. [1]

It sounds like a language model has learned how to optimize another language model by reading its brain.

That is an appealing headline, but it hides the more important engineering question: **did the edit add mathematical reasoning, or did it repair a failure that prevented an existing capability from producing an answer?**

The answer matters because these are different forms of progress. One improves what a model knows or can compute. The other fixes how the computation unfolds. The second may be easier to localize, easier to edit, and easier to verify. It may also be the first practical application of what the authors call **Weight-Space Programming**.

This tutorial investigates that idea from first principles. We will read the eight released experiments, inspect the boundaries of the claims, build actual networks by writing their PyTorch tensors directly, audit a sparse repair, and test whether our intervention survives a distribution shift. The companion repository contains the working implementations, benchmark results, statistical controls, tests, and publication-ready figures.

**Important distinction:** the original Qwen results come from the MIT CSAIL researchers. Our Python workshop is an independent educational experiment using deliberately constructed tiny networks. It does not call Astra, download Qwen3.5-0.8B, reproduce MATH-500, or establish that an ordinary LLM can perform the same optimization.

**Primary research:** [Sparks of Weight-Space Programming, Takehi et al., November 2026](https://rikiyat.github.io/blog/weight-space-programming/). [Released prompts, edits, logs, and evaluation artifacts](https://github.com/RikiyaT/lm_understand-logs). [2]

---

## Table of contents

1. What Weight-Space Programming actually means
2. Reading a neural network from raw numbers
3. The first three experiments: games, diffusion, language models
4. Why weights are not uniquely readable
5. Writing a neural network without training it
6. A network that plays Breakout after one reply
7. The one-shot car repair and its architectural prior
8. Go: from weight generation to feedback-guided edits
9. Understanding the Qwen3.5 math result
10. What the 131,072 edited locations really mean
11. Connection to projections and low-rank model editing
12. The sycophancy failure and the limits of weight-only interpretation
13. What the released harness does and does not permit
14. Build our independent PyTorch laboratory
15. Experiment A: compile XOR directly into weights
16. Experiment B: read a function from weights
17. Experiment C: break weight-reading with hidden-unit permutations
18. Experiment D: repair a deliberately corrupted policy
19. Experiment E: compare against non-LLM controls
20. Experiment F: distribution shift and an independent rollout
21. Implement a controlled projection edit
22. Audit edits without trusting the editor
23. Five-seed results and their limitations
24. Measuring actual reasoning gains versus completion gains
25. Better baselines for future research
26. A rigorous evaluation protocol for weight editors
27. Applications and security implications
28. Where this research is still weak
29. How to run and extend the companion lab
30. What Weight-Space Programming changes
31. References and further reading

---

## 1. What does it mean to program a neural network's weights?

Today, nearly all machine learning engineers interact with parameters indirectly. We specify an architecture, define a loss, choose an optimizer, and let gradient descent update tensors. If we want a model to behave differently, we normally change the training examples, the objective, the reward, the prompt, an adapter, or the inference-time computation.

Weight-Space Programming proposes a different interface:

**Describe the desired behavior, inspect the existing parameters, and deliberately write new parameter values.**

This does not mean the neural network stops being a collection of matrix multiplications. It means an external reasoning system constructs or modifies the matrices themselves, analogous to a programmer writing an implementation rather than asking an optimizer to search for one.

The authors investigate three related abilities.

**Weight reading.** Given a network's raw parameters, infer which behavior or computational circuit they implement. This resembles reverse engineering an executable whose source code is missing.

**Weight writing.** Given the architecture and the desired behavior, populate parameter tensors so the network computes the intended function. This resembles compiling a specification into a fixed neural circuit.

**Weight repair.** Given a failing network and evidence about its failure, identify the part of the computation causing it and edit a small subset of weights. This resembles patching an existing program while preserving everything else.

These abilities sit at different points on a difficulty spectrum. Handcrafting a two-layer ReLU network to compute XOR is approachable. Interpreting a trained network with many symmetries is harder. Editing a large pretrained language model without degrading its other behaviors is harder still.

The MIT work is noteworthy because a general-purpose frontier language model is doing some of this reasoning without a specialized weight-prediction model trained for the task. The experiment does not eliminate computation or prior knowledge. The agent receives architectural descriptions and, for some experiments, a carefully constrained tool environment. Those details matter. [1, 2]

## 2. What can a weight matrix tell us?

Start with the simplest possible feed-forward network. It takes an input vector, multiplies it by a matrix, applies a nonlinearity, then multiplies by another matrix.

Consider a classifier with four input features, sixteen hidden neurons, and three output actions.

```text
Input x: four numbers
        |
        v
First weight matrix: 16 by 4
        |
        v
ReLU activation: 16 numbers
        |
        v
Second weight matrix: 3 by 16
        |
        v
Three action logits
```

Every row of the first matrix detects a particular weighted combination of input coordinates. Every column of the second matrix describes how that hidden activation influences the output actions.

In a small, sparsely wired network, these relationships can be readable. If one hidden neuron receives almost exclusively a signal describing paddle position, while another tracks ball movement, the first layer can reveal something about the controller's purpose. If the action head gives strong positive and negative weights to left and right actions, it can support that interpretation.

But notice the conditions we have already smuggled in: we know what the input coordinates mean, we know the output action encoding, and we expect hidden features to occupy unusually legible positions.

A trained model is under no obligation to organize itself this way. Its hidden states may distribute one useful concept across many neurons, and several equally functional parameter settings may look entirely different.

This observation will become the first failure mode in our lab. It is also why the strongest experiments in the paper are not the visually obvious small networks, but the ones that reach into middle-layer computations and identify mechanisms without input or output embeddings.

## 3. What the original experiments actually found

The study presents eight main experiments. They are not all demonstrations of the same capability, and they should not be collapsed into a single claim that the model can magically read any neural network. [1]

### Experiment 1: identifying games from policy weights

The researchers trained same-architecture policies for Breakout, Pong, Pac-Man, and Freeway. Each policy used the same input and output layouts. Astra was shown raw weights, not gameplay or rollout feedback, and correctly named each of the four games.

This matters because the experiment reduces easy shortcuts from differences in architecture or action count. However, the model still had access to the meaning of the input channels and to architectural context. Its interpretation relied on patterns such as strong spatial weights near the location of a paddle, and action output connections that differentiated horizontal movement.

Four correctly identified networks are interesting evidence of capability. They are not yet a large, statistically powered weight-reading benchmark with unseen architectures, many training seeds, or adversarially reparameterized networks.

### Experiment 2: describing what a diffusion model generates

Three small diffusion models were trained to generate circles, squares, or triangles. Astra was asked to inspect their weights without a forward pass and describe the generated shape.

The model could identify the shapes, but the report includes a useful mistake: it described their *outlines* even when the generators produced filled shapes. That error exposes the boundary between reading a recognizable geometric motif from output weights and reconstructing the actual output distribution of the model.

Interpretability is often most revealing when the system is wrong in a structured way. A plausible explanation can be locally sensible while still missing the complete computation.

### Experiment 3: identifying languages from middle Transformer layers

The researchers trained three small character-level Transformers, each with approximately 7.2 million parameters, on Python code, English, or Japanese rendered in romaji. They removed the input embeddings and output layers, leaving only the four internal Transformer blocks.

Astra could analyze the weights with Python but was forbidden to run the model or its layers on text. It identified all three language domains and also located the copying heads that the researchers later measured independently. The reported copies matched the ground-truth head lists in those controlled tests.

The researchers suggest that a higher concentration of induction-like behavior can signal code, where repeated variable names and recurring strings are common. They also describe distinctive output-direction structure in the romaji-trained model. [1, 3]

This is a much more interesting form of weight reading than noticing that a first-layer filter resembles a paddle. It attempts to infer a behavioral property from composition of internal weight matrices.

But even here, the number of trained models is small. The experiment is better understood as a compelling demonstration of feasibility than as evidence that arbitrary hidden circuits can reliably be identified by a frontier LLM.

## 4. A neural network's function does not have a unique set of weights

Before writing code, we need one theoretical obstacle that changes how these experiments should be evaluated.

Suppose a network has sixteen ReLU hidden units. If you swap hidden units 3 and 7, then swap the corresponding outgoing weight columns, its output remains exactly the same. The input-output function has not changed. Only the storage positions have changed.

You can also multiply a ReLU neuron's incoming weights and bias by a positive constant, and divide its outgoing weights by that constant. Because ReLU is positively homogeneous, the function is preserved.

These are **parameter symmetries**. They create many different-looking parameter arrays that implement the same computation.

The consequence is straightforward: a language model that claims a particular hidden index is inherently meaningful may be learning artifacts of the training setup, rather than the underlying function.

For an interpretability method to deserve trust, it should remain accurate under function-preserving transformations, or explain precisely which coordinate system it relies on.

For more complicated networks, the issue gets deeper. Attention heads can be permuted within layers under corresponding parameter transformations. Hidden spaces can admit additional changes of basis in some linear subcomputations. Quantization, fine-tuning and model merging further change the numerical appearance of the parameters.

**The object we want to understand is the computation. The tensor values are one representation of that computation.**

In our lab, this difference becomes measurable: a brittle first-neuron reader identifies the correct synthetic task only about 38% of the time after random hidden-neuron permutations, whereas a reader designed to be invariant to those permutations reaches 100% on the same deliberately simple tasks. Neither approach is a learned model or a general weight-understanding system.

## 5. A tiny neural network can literally be written like a program

The first coding experiment removes data and optimization entirely. We will write six numbers into a small PyTorch network so it implements XOR.

XOR returns one when its two binary inputs differ, and zero otherwise. A linear classifier cannot implement XOR because its positive and negative examples are not linearly separable. A two-layer ReLU network can.

Here is the complete circuit:

```python
import torch
from torch import nn

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

net = XORNet()

with torch.no_grad():
    net.fc1.weight.copy_(torch.tensor([
        [ 1.0, -1.0],
        [-1.0,  1.0],
    ]))
    net.fc2.weight.copy_(torch.tensor([[1.0, 1.0]]))

inputs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

print(net(inputs).flatten())
# tensor([0., 1., 1., 0.], grad_fn=...)
```

The first hidden unit computes the positive part of the difference between the two inputs. The second computes the positive part of the difference in the other direction. Their sum is the absolute difference for binary inputs, which is exactly XOR.

We did not find these weights by gradient descent. We compiled a known function into a neural architecture. This is a genuine example of programming weight space, although it is a very easy one because we designed the entire circuit.

A compiler that knows how to generate this circuit does not need a language model. The relevant question for frontier systems is whether they can discover useful circuit constructions for less obvious tasks, especially when existing weights and trained computations need to be preserved.

Our companion repository includes this example in `weightlab/core.py`, with a test that checks all four inputs and confirms that the network contains exactly six parameters.

## 6. The original study goes far beyond XOR

In the fifth experiment, the researchers give Astra the rules and source code for three MinAtar games and a neural-network architecture whose weights start at zero. Astra writes new weights in natural-language-driven fashion, without gradient training or game rollouts during the one-shot generation.

The reported scores are particularly dramatic for Breakout: the handcrafted network scores **517.4**, compared with **15.8** for PPO after 100 million environment steps and **25.3** after 500 million steps. On Seaquest, Astra's **53.1** is above the reported 100-million-step PPO baseline but below the 500-million-step score of **88.9**. On Asterix, **53.2** is similarly near, but below, the longer-trained PPO result of **55.2**. [1]

The mechanism is more enlightening than the score ratio. Astra interprets the game rules and expresses an algorithm in network layers. For Breakout, the researchers describe units that estimate where the ball will land, including simple wall reflections, and select actions based on the difference between the landing position and paddle position.

This is a form of **neural program compilation**, not conventional reinforcement learning. A structured symbolic algorithm is transferred into parameters that happen to live inside a neural-network architecture.

That is a powerful idea, but it is not automatically a fair victory over the general-purpose learning algorithm. Astra receives a rich description of the environment's rules, can reason over the source code, and encodes a hand-designed strategy. PPO must infer useful behavior from reward and experience under its own optimization procedure. The comparison establishes that a good supplied inductive bias can outperform a particular learned policy in these environments. It does not establish that one-shot weight programming dominates PPO in general.

The strongest next experiment would compare Astra with classical hand-coded algorithms compiled into exactly the same architecture. A simple, well-specified controller might be an equally strong or stronger baseline.

## 7. The car repair: patching an obstacle rather than replacing a controller

Experiment 4 demonstrates a second type of weight programming. A PPO policy in the Kinetix physics environment drives toward a ramp but repeatedly gets stuck. The needed strategy is to reverse first, accumulate enough distance, then accelerate forward.

Astra receives the existing weights, a scene description, and one recorded failure. The researchers report that it changes **6,073 of 171,664 parameters**, approximately **3.5%**, in one tool-free reply. The modified policy succeeds in **1,024 of 1,024** reported evaluation episodes. [1]

The generated fix does not replace the entire trained policy. It adds a small route through selected internal units that detects the relevant state, triggers a reverse-and-accelerate maneuver, and then lets the original network take over again.

This resembles a software hotfix: keep the stable functionality, add an override for the failing region, and preserve the rest of the execution path.

That last clause is vital. A good patch must not only improve the measured failure case. It must avoid breaking nearby cases, and it must generalize beyond the exact recorded trace. A 100% success rate across 1,024 evaluation rollouts supports the local claim, but an engineering evaluation should still ask whether initial conditions, physics parameters, and out-of-distribution variants were varied sufficiently.

Our companion benchmark tests the same *principle* in a much smaller synthetic controller. We deliberately introduce an unproductive action pathway, then repair exactly two offending weights. We compare the repair with random changes and with the surprisingly competitive baseline of simply zeroing the two largest output weights.

## 8. Go demonstrates what limited forward-pass feedback changes

Astra is then asked to write a Go-playing neural network with roughly **1.48 million parameters**. The input represents a 9-by-9 board, and the outputs represent board moves and passing. The authors permit a sandbox for reading and writing weight tensors and allow budgets of 0, 10, 25, or 50 forward-pass calls.

A forward-pass call can include batches of game evaluations. Therefore, **50 forward passes is not 50 individual state evaluations**. It is a budget of controlled feedback opportunities, each potentially containing substantial evaluation computation.

The proposed weights implement Go-specific operations: identifying liberties, tracking stone groups, finding captures or threatened groups, and scoring legal moves. With zero feedback, the networks already perform well against two smaller PPO opponents. With additional feedback, the system revises targeted pieces of its neural program and ultimately beats the reported equal-size PPO comparison trained on 20 million games. [1]

What changed between the zero-feedback and 50-call conditions?

The first is pure one-shot compilation from known rules. The second is an iterative development loop. The LLM proposes code or weight edits, invokes a controlled experiment, receives evidence, and revises its hypothesis.

It remains gradient-free, but it is no longer **feedback-free**. The distinction is fundamental. Many gradient-free methods can improve policies with evaluations. A fair comparison must equalize evaluation budgets, available prior knowledge, and total computational cost.

This leads us to the experiment that generated the headline.

## 9. Qwen3.5-0.8B: when the model's problem is that it cannot stop

The final experiment starts with an unusual debugging symptom. During practice runs, the base Qwen3.5-0.8B model keeps generating reasoning instead of reaching a final mathematical answer. The researchers report that **85 of the first 96 practice answers reached the 32,768-token limit**. [1]

Qwen's own model card independently warns that its 0.8B model can enter thinking loops under some recommended sampling settings. It recommends tuning sampling parameters and monitoring generation so loops can be detected and interrupted. That warning matters when interpreting the experimental baseline. [4]

Astra is given a Python environment where it can inspect and modify Qwen's tensors. Unlike the first experiments, it can also request limited forward-pass evaluations on practice questions drawn from the GSM8K and MATH training sets. The authors allocate budgets of 10, 25, and 50 such calls, each allowed to contain a batch of examples. Gradients and conventional training are not allowed.

The system develops the hypothesis that the failures arise from a small number of directions being repeatedly written into the residual stream by late MLPs. It ultimately modifies 64 columns in each of two output projection matrices, one associated with layer 15 and one with layer 19. The addressed footprint is 131,072 parameter locations. [1]

With the 50-call edit, the reported results are:

- **MATH-500:** 7.9% to 48.1%.
- **GSM8K:** 13.8% to 67.1%.
- **IFEval:** 21.0% to 62.3%.
- **MATH-500 generations hitting the token limit:** 93.0% to 33.1%.

The 10-call and 25-call variants deliver smaller gains and retain high token-limit rates. The effect is not a smooth linear function of feedback budget. The effective correction appears after enough information has accumulated to locate and adjust the relevant direction.

This is an intriguing result, but it should be described precisely. A metric such as MATH-500 accuracy combines at least two issues: whether the model can solve the mathematical problem, and whether it successfully produces an evaluable answer before termination. If the generation process stalls, the benchmark can score a problem incorrect even if the model's intermediate computation contained useful partial reasoning.

In other words, **the measured outcome can improve without an equally large increase in the model's underlying mathematical knowledge**.

The authors explicitly investigate that distinction in a follow-up. They ask Astra to improve Gemma 3 270M, which does not suffer from such severe repetitive reasoning. Its reported MATH-500 performance remains near the original level, roughly 6.8% to 7.7% across feedback budgets. In this case, direct weight editing did not manufacture a substantially better mathematician. [1]

The Qwen repair remains impressive. But it is an example of a localized operational failure whose removal can expose existing capability, not a demonstration that arbitrary missing skills can be installed by editing a few weights.

## 10. What did 0.016% of the weights actually mean?

The distinction between **addressed locations** and **bitwise changed tensor elements** is worth making explicit.

The original post describes edits targeting 131,072 weight locations. A separate technical audit of the released checkpoints reports that **130,188 tensor elements differ bitwise**, while the remaining 884 addressed locations round back to their original values. The audit also notes that the original 0.016% calculation uses the rounded 0.8-billion model size rather than an exact parameter denominator. [1, 5]

This is not a contradiction about the existence of the patch. It is a measurement-definition issue, familiar to engineers who have compared numerical checkpoint files:

- The algorithm selected a set of locations to update.
- The computation produced floating-point values for those locations.
- The checkpoint stored finite-precision values.
- Some attempted updates were too small to change the stored bits.

For reproducibility, all of these quantities should be reported separately. A claim about the fraction of a model that was edited is incomplete until it specifies its denominator, numerical representation, and threshold for counting an element as changed.

Our lab's `diff_audit()` function returns the exact number of changed tensor entries, their locations by parameter name, the total parameter count, and the maximum absolute update. It makes no approximation based on how many indices an editing algorithm *intended* to touch.

The independent checkpoint audit is not an independent rerun of the complete Qwen benchmark. Its author reports checking the released checkpoint difference and re-grading reported answers, but not executing Qwen for a new end-to-end evaluation. That is an important boundary when discussing independent confirmation. [5]

## 11. The weight-space operation is a projection, not a magical neuron switch

A Transformer MLP contains matrices that map hidden activations back into the residual stream. If a vector in that output space corresponds to an undesirable behavior, one possible intervention is to suppress the component of particular output columns along that direction.

Suppose `W` is a two-dimensional weight matrix with shape `[output_dimensions, input_dimensions]`, and `u` is a normalized direction in its output space. For a selected column, calculate how much its output vector projects along `u`, and subtract a chosen multiple of that component.

```python
import torch

@torch.no_grad()
def suppress_output_direction(weight, direction, columns, scale=1.0):
    """Return a copy with a direction suppressed in chosen columns."""
    result = weight.detach().clone()
    unit = direction / direction.norm()
    block = result[:, columns].clone()
    projected = unit @ block
    result[:, columns] = (
        block - scale * unit[:, None] * projected[None, :]
    )
    return result
```

At strength 1.0, this operation removes the selected components along the direction. At strength 0, it changes nothing. At strength greater than 1, it goes beyond removal, reversing and amplifying that component in the opposite direction.

A technical audit of the released Qwen checkpoint reports that the implemented scale is approximately 6, so the edited component is not merely projected away. It is reversed and amplified relative to the original. [5]

That detail is easy to overlook because phrases such as *remove the loop direction* sound like deleting one stored concept. Real matrix edits can have more complicated effects. They alter the contribution of particular pathways to future activations. In a deep nonlinear network, downstream layers may transform these changes in ways that are not obvious from the local matrix operation.

This is closely related to work on rank-one model editing and feature-level intervention. The ROME paper, for example, studied targeted modifications to factual associations in GPT. The 2024 refusal-direction work demonstrated that certain behavioral tendencies can be influenced through low-dimensional intervention, although subsequent research suggests that a single-direction account can be incomplete. [6, 7, 8]

A weight patch is not proof that a behavior is stored only where the patch was applied. It tells us that changing that pathway is causally sufficient to alter the measured outcome under the tested conditions.

## 12. The failed sycophancy test may be the most useful negative result

The research also investigates whether raw weights alone reveal a learned preference to agree with users even when their suggestions are wrong.

Two versions of Gemma 3 270M are fine-tuned with similar procedures. One is taught to give correct answers. The other is taught to agree with the user's proposed answer. To remove easy signals, the researchers freeze the input embeddings and output layers during the fine-tuning process.

Astra is then asked to identify the sycophantic model using only the weights. It fails in all four attempts reported in the post. It can find numerical differences, but it cannot reliably map those differences to the targeted behavior. [1]

Only when the experiment allows a limited additional observable, **attention patterns on prompts that Astra constructs**, does the system successfully identify the sycophantic model across the three reported pairs.

Why is this failure so informative?

It demonstrates that identifying a numerical difference is not the same as understanding its behavioral consequence. Two neural networks can have millions of differing parameter values and still disagree mainly on a narrow semantic property. Conversely, a seemingly small edit can change behavior dramatically. Without a causal model or carefully chosen observable, a weight difference can be underdetermined.

The paper's result suggests a continuum rather than a binary choice between weight-only interpretation and black-box testing:

1. Read the weights directly.
2. Build a structural hypothesis about the computation.
3. Choose a minimal intermediate observable that discriminates between competing hypotheses.
4. Make a causal edit.
5. Evaluate independently on held-out conditions.

That workflow is closer to scientific debugging than merely asking a model to describe a matrix.

## 13. What does the released agent harness actually allow?

The authors released logs for all eight experiments. The repository records prompts, model outputs, tool-call artifacts, networks, evaluations, and per-run metadata. It is valuable because readers can inspect concrete evidence instead of relying only on the article's screenshots. [2]

The setup distinguishes built-in Codex tools from tools supplied by the research harness. The released README says the researchers disabled shell access, browser access, native file editing, plugins, sub-agents, and other built-in Codex tool interfaces. They audited all 1,125 released turns for this condition.

However, Experiments 3, 6, 7, and 8 used Python through an external orchestration loop. Astra generated Python snippets, the researchers ran them in their own restricted environment, and printed outputs were returned to the conversation.

Therefore, the accurate description is not *the model had absolutely no tools or external computation*. It is *the model's built-in tools were disabled, and certain experiments allowed a separately constrained, instrumented Python tool with specified limits*. [2]

This is a reasonable scientific design. It gives the researchers a way to constrain forward passes, disable network access, inspect code, and count evaluation opportunities. But it matters when comparing the result with a no-feedback or no-code baseline.

The released logs do not include every earlier prompt-development trial. The repository explicitly states that earlier prompt versions are omitted. This is not unusual, but it means the public record should not be interpreted as a complete account of every attempt made during the research process.

For large-scale comparisons, the full development and selection budget becomes another variable worth measuring.

---

# Building an independent weight-space programming lab

## 14. What this lab will and will not reproduce

Our goal is not to pretend that a laptop experiment can reproduce an Astra run over almost a billion parameters. Instead, we isolate three mechanisms and make their assumptions visible.

**First, we compile a Boolean function into weights without an optimizer.** That shows what writing a neural program actually looks like.

**Second, we read a task identity from the weights of a known network and perturb the parameter representation while preserving the function.** That tests whether a weight reader is discovering computation or memorizing a convenient neuron index.

**Third, we deliberately corrupt a controller with an unproductive action pathway.** We then compare a semantic, sparse weight repair against a partial repair, a random equal-budget edit, a magnitude-pruning baseline, an incorrect-path edit, and a global rescaling control.

Finally, we evaluate the candidate weights on unseen samples, a nuisance-channel distribution shift, and a simple closed-loop navigation environment. The evaluator does not use the editing policy's self-reported claims.

The entire benchmark contains no LLM invocation. The policy for each edit is explicitly programmed. This makes it unsuitable as evidence of autonomous weight-space reasoning, but very suitable for testing what strong evaluation controls should look like.

### Repository layout

```text
weight_space_programming_lab/
├── weightlab/
│   ├── core.py         # Known neural circuits and weight readers
│   ├── editing.py      # Sparse edits and exact tensor-diff audit
│   ├── evaluate.py     # Held-out classification and environment rollouts
│   ├── projection.py   # Directional output-matrix projection
│   ├── demo.py         # Interactive examples
│   └── benchmark.py    # Multi-seed ablations
├── tests/
│   └── test_weightlab.py
├── results/
│   └── benchmark.json
├── figures/
│   ├── 01_sparse_edit_heldout.png
│   ├── 02_weight_reading_symmetry.png
│   └── 03_distribution_shift.png
├── audit_safetensors.py
├── plot_results.py
├── requirements.txt
├── BENCHMARK_NOTES.md
└── README.md
```

The default code requires PyTorch and can run on CPU. It does not need pretrained checkpoints, model credentials, a GPU, or an online evaluator.

## 15. Experiment A: writing six numbers into a working neural circuit

The full XOR implementation appeared earlier. It is worth explaining why it is more than a party trick.

A conventional model learns a function indirectly. We allocate a parameterized hypothesis class, show it examples, compute a loss, and search for weights that minimize that loss. In this example, we instead reason about the desired computation, choose a representation, and fill each element of the weight matrices deliberately.

The first ReLU detects one directional difference. The second detects the opposite. Together they compute absolute difference. Because the domain is two binary inputs, absolute difference exactly implements XOR.

The interesting part is that the output is still an ordinary `torch.nn.Module`. After construction, inference does not know or care that the weights came from a programmer rather than an optimizer.

This is also an appropriate place to identify the boundary of the demonstration. We chose a function with a known analytic solution and an architecture that expresses it cleanly. For real game policies, the mapping from game rules to a neural circuit is much less obvious. For pretrained language models, the challenge is not only producing a correct function but modifying a densely connected, already useful function without damaging unrelated behavior.

To run the complete experiment:

```bash
python -m weightlab.demo
```

The first printed result shows all four XOR inputs and their exact outputs. A unit test checks the truth table, so the example cannot succeed merely by formatting a plausible-looking answer.

## 16. Experiment B: a transparent weight reader

Our second network has four input coordinates and three actions. For a given task, the sign of one input coordinate determines which of two productive actions should be selected. A separate nuisance coordinate drives an intentionally unhelpful third action in the corrupted version.

In the clean network, two ReLU units encode the positive and negative parts of the task coordinate. Other hidden units are either dormant or reserved for the nuisance signal.

```python
import torch
from weightlab.core import make_policy

model = make_policy(axis=2, nuisance=3, broken=False)

print(model.fc1.weight.shape)  # [16, 4]
print(model.fc2.weight.shape)  # [3, 16]
```

A weight reader can inspect which input coordinate has the largest aggregate absolute incoming weight:

```python
def inspect_axis_invariant(model):
    score = model.fc1.weight.abs().sum(dim=0)
    return int(score.argmax().item())
```

For this toy architecture, the useful task coordinate has a clear signature. If the task is controlled by input coordinate 2, the function reports 2. It does this from the parameters alone, without running examples through the network.

That sounds similar in spirit to the original game-reading experiment, but it is far simpler. We authored the relevant circuit, know the input layout, and are not trying to infer an unseen task from a learned representation. The example establishes the method's mechanics, not the ability of a language model to reverse-engineer arbitrary networks.

The next experiment is more revealing: we keep the function unchanged while altering where it is stored.

## 17. Experiment C: changing every important neuron index without changing the behavior

Suppose the first two hidden units carry the sign of the task signal. A naive inspector might look at the first unit and report whichever input coordinate has its largest coefficient.

```python
def inspect_axis_naive(model):
    return int(model.fc1.weight[0].abs().argmax().item())
```

That heuristic works for the original hand-compiled layout, but it depends on an arbitrary naming convention. If we permute all sixteen hidden units and permute their outgoing columns in the matching way, the network's outputs remain unchanged.

```python
import torch
from weightlab.core import make_policy, permute_hidden

m = make_policy(axis=2, nuisance=3)
g = torch.Generator().manual_seed(7)
order = torch.randperm(16, generator=g)
remapped = permute_hidden(m, order)

x = torch.randn(64, 4, generator=g)
assert torch.allclose(m(x), remapped(x))
```

Both networks implement the same input-output mapping. The only change is their hidden-unit labels.

Our benchmark generates networks for three task coordinates, makes sixty random hidden-unit permutations for each, and repeats the entire process for five random seeds. This gives **900 function-preserving representation variants** in total.

The brittle first-unit reader identifies the correct task **38.11%** of the time on average. The aggregate absolute-weight reader reaches **100%**.

The latter result is not an argument that this simple reader will work on realistic pretrained models. It is an argument that **testing invariance is mandatory before claiming that a weight-only interpretation is semantically meaningful**.

A useful follow-up would apply more sophisticated function-preserving transformations: positive per-unit rescaling, permutations within attention heads, and any change-of-basis transformations supported by the architecture. A proposed language-model weight reader should be evaluated on all of them, not only on the original parameter ordering.

## 18. Experiment D: constructing a policy that fails for one narrow reason

We now build a toy controller with three possible actions. Two actions make progress toward a goal. The third action does nothing.

The controller receives a four-dimensional observation. Its first coordinate encodes the position relative to the target. The fourth coordinate is irrelevant nuisance information. An ideal controller should ignore the nuisance channel and move toward the target.

The clean, hand-compiled network has two useful output connections. The corrupted version adds two stronger links from nuisance-activated hidden units into the unproductive action.

```python
from weightlab.core import make_policy
from weightlab.evaluate import classification_accuracy

healthy = make_policy(broken=False)
corrupted = make_policy(broken=True)

print(classification_accuracy(healthy, seed=20))
print(classification_accuracy(corrupted, seed=20))
```

The corruption is deliberately obvious when one knows the designed circuit. The first hidden layer contains one pair of units for positive and negative target position, and another pair for positive and negative nuisance values. The output layer maps the nuisance pair strongly into the stall action.

In mathematical terms, the network's productive logits depend on the sign and magnitude of the target coordinate, while the unproductive logit grows with the absolute value of the nuisance coordinate. This introduces a spurious competition for the winning action.

On our held-out synthetic samples, the corrupted network selects the correct productive action only **29.83%** of the time. Under a shift that multiplies the nuisance coordinate by 2.5, accuracy falls to **12.31%**.

This is analogous to a model that already has useful task behavior but repeatedly takes an unproductive branch. The analogy is deliberately limited: a three-action policy is not an autoregressive reasoning model, and its stall logit is not a learned Qwen reasoning-loop circuit.

## 19. Experiment E: remove precisely two offending weight connections

The semantic repair knows the input and output meanings. It does not know any test labels. It simply removes the two links that carry the nuisance signal into the unproductive action.

```python
import copy
import torch

@torch.no_grad()
def semantic_repair(model):
    fixed = copy.deepcopy(model)
    fixed.fc2.weight[2, 2] = 0.0
    fixed.fc2.weight[2, 3] = 0.0
    return fixed
```

Those indices are not discovered autonomously by an LLM. They are selected because we authored the network and understand which hidden paths are unwanted. This is a controlled illustration of a semantic weight patch.

After applying the repair, our three-action classifier returns to 100% on the held-out synthetic test, because it is now functionally identical to the original correct controller. The edit changes only **two of 131 stored parameters**, roughly **1.53%** of this tiny network.

This is not evidence that repairing two weights is sufficient in real models. It tells us what a successful surgical patch should look like when the correct cause is already known.

But a central question remains: would a simple optimizer-free alternative achieve the same result?

### The non-LLM control that also works

The corrupted links are also the two largest-magnitude output weights. A magnitude-pruning heuristic that removes the two largest entries in the output matrix finds exactly the same correction.

```python
from weightlab.editing import propose_edit

corrupted = make_policy(broken=True)
semantic = propose_edit(corrupted, "semantic")
magnitude = propose_edit(corrupted, "top_magnitude")

assert torch.equal(
    semantic.fc2.weight,
    magnitude.fc2.weight,
)
```

Both approaches restore **100%** accuracy. This is the single most important cautionary result in our lab. A successful patch is not evidence of sophisticated reasoning unless it beats simple alternatives under fair controls.

In the original Qwen experiment, a corresponding set of controls could include activation-outlier suppression, low-rank projections selected by conventional statistics, magnitude pruning, targeted sampling changes, and equal-budget black-box optimization. The question would not simply be whether Astra's edit works, but how much value Astra adds beyond the best cheap method.

## 20. A matched-budget random edit and an intentionally incomplete repair

Our random control zeroes **two randomly selected nonzero output connections**. This is important: choosing two entries from the entire matrix would mostly select zeros, producing no actual edit and giving an unfair comparison.

The matched control changes exactly two nonzero weights, but has no knowledge of which pathways are productive or unproductive.

Across five seeds, this random two-weight edit reaches **37.97%** familiar accuracy and **35.81%** accuracy under the shifted nuisance distribution. It is somewhat better than the corrupted model in some conditions, and worse in others, but it does not reliably restore the policy.

The partial semantic repair removes only one of the two bad connections. It reaches **64.83%** familiar accuracy and **55.88%** accuracy under shift. In the navigation environment, it often succeeds because repeated action opportunities can compensate for some stalls. That is a reminder that one-step classification accuracy and end-to-end task success are different metrics.

Two further controls help check the causal interpretation.

**Global shrinkage.** Multiplying all output weights by a constant changes numerical logit magnitudes, but not their ordering. Accuracy therefore remains unchanged in the default deterministic argmax policy. A large numerical change need not cause a meaningful behavioral change.

**Wrong-path editing.** Removing the two productive output links instead of the nuisance links destroys the correct mapping. Held-out classification accuracy drops to 0% in this constructed setup. A sparse edit is not automatically a good edit merely because few parameters changed.

All of these controls appear in the saved benchmark JSON and can be reproduced using the same evaluator.

## 21. Independent verification with rollouts and distribution shift

A scientific result is not complete when the editing code prints `success = True`. The evaluator must measure what the modified network actually does under conditions it did not see during construction.

Our benchmark includes two forms of held-out evaluation.

**One-step classification.** The evaluator draws fresh random observations, computes the correct productive action from the target-coordinate sign, and compares it with the network's argmax. A shifted test multiplies the nuisance coordinate by 2.5.

**Closed-loop navigation.** The evaluator starts an agent at random positions along a one-dimensional axis. The controller repeatedly selects left, right, or stall. Success means entering a small region around the target before the episode budget expires.

```python
from weightlab.core import make_policy
from weightlab.editing import propose_edit
from weightlab.evaluate import navigation_success

broken = make_policy(broken=True)
fixed = propose_edit(broken, "semantic")

print(navigation_success(broken, seed=202))
print(navigation_success(fixed, seed=202))
```

Across five seeds, the broken controller succeeds in **30.0%** of the navigation episodes, compared with **100%** after the semantic repair. The matched magnitude baseline also reaches 100%, while random two-link edits average **50.8%**.

The success test is not a claim about the difficulty of real-world robotics. Our environment is intentionally small and deterministic apart from randomized observations and starting states. The point is to prevent the editor from granting itself a passing grade.

A serious weight-editor benchmark would extend this idea with separate evaluation code, hidden seeds, withheld environment variants, and checkpoint integrity constraints. It should also prohibit an editing agent from accessing those hidden test definitions or mutating the evaluator.

## 22. A projection edit, with a conservation test

We also implement a general low-rank direction suppression operator in `weightlab/projection.py`. It is independent of the three-action repair and does not pretend to reproduce the exact Qwen patch.

For a matrix `W` and output-space direction `u`, the operator updates a selected set of columns. At scale 1.0, each selected output vector becomes orthogonal to the chosen direction, up to floating-point precision. Every unselected column remains unchanged.

The tests verify all three properties: correct projected direction, unchanged unselected columns, and no mutation of the original tensor. An additional test confirms that using scale 6.0 flips the original projection component to approximately negative five times its initial value.

Why test these algebraic invariants? Because an edit's scientific interpretation depends on what the code *actually does*. A local patch can be described as a projection while its implementation instead transposes a matrix, edits the wrong axis, normalizes an empty vector, or changes tensors in place. Any one of these would invalidate a mechanistic claim.

The released Qwen checkpoint invites a similar audit: enumerate affected tensors, count exact changed elements, inspect which rows or columns were touched, compute projection coefficients, compare file hashes, and verify that the saved checkpoint implements the proposed update. [2, 5]

Our optional `audit_safetensors.py` accepts a pair of single-file `safetensors` checkpoints and counts differing tensor elements without loading untrusted Python pickle objects. It does not run either model or assess task performance. That is intentionally a separate stage.

## 23. Five-seed benchmark results: where editing helps and where it does not

The complete experiment uses five deterministic seeds. Each seed runs 3,000 one-step classification cases per policy, 3,000 shifted cases per policy, and 300 independent navigation episodes per policy. For the weight-reading experiment, it generates 60 hidden-neuron permutations for each of three tasks. All outputs, per-seed records, and summary statistics are saved to `results/benchmark.json`.

The three most important results are:

**Weight reading is sensitive to parameter symmetries.** The first-neuron heuristic reaches 38.11% correct task identification, compared with 100% for the aggregation-based invariant reader on the same synthetic representations.

**A small, correctly diagnosed edit can completely repair a simple failure.** Classification accuracy rises from 29.83% to 100%. Under nuisance shift, it rises from 12.31% to 100%. Closed-loop success rises from 30.0% to 100%.

**Sophisticated interpretation is not always needed.** Magnitude pruning matches the semantic repair exactly in this constructed case. The intended lesson is not that weight programming is useless. It is that every claimed benefit must be measured against simple interventions that operate on the same weight space.

These outcomes follow from the architecture we deliberately designed. They are not evidence that the same performance changes would occur on an unknown trained Transformer. The models differ radically in scale, the task semantics are supplied, no LLM searches for the correction, and the distributions are synthetic rather than natural-world data.

The plots in the companion repository make the comparisons easy to inspect:

**Figure 1.** Sparse-edit held-out accuracy. Compares the corrupted model, semantic edit, magnitude pruning, random two-weight control, partial repair, and wrong-path intervention.

**Figure 2.** Weight-reading accuracy before and after making the interpretability method respect hidden-neuron permutations.

**Figure 3.** Familiar versus nuisance-shift accuracy for several editing strategies.

The figures are generated directly from the saved JSON. They are not recreated by hand from values copied into a charting script.

## 24. The hardest metric: did the model learn more, or merely finish more often?

Suppose a model has an internal reasoning procedure that can solve some fraction of a task distribution. However, it also has a termination pathology: even after reaching useful partial conclusions, it sometimes repeats the same derivation until generation times out.

A repair that fixes only the termination pathology can produce a large increase in benchmark accuracy while leaving the model's mathematical representation and basic inference circuits mostly unchanged.

This is neither trivial nor dishonest. It is an important operational improvement. A language model that cannot finish an answer is much less useful than one that can. But researchers should measure the improvement in a way that identifies its cause.

For model-editing studies, I would report at least the following quantities separately:

- The fraction of prompts for which the model produces a valid final answer.
- Accuracy on the original benchmark, including failed and incomplete generations.
- Accuracy restricted to completed generations, with the selection bias clearly acknowledged.
- Average and high-percentile output lengths.
- The frequency of repeated spans, cyclical reasoning, and token-limit termination.
- Accuracy at multiple generation caps, especially shorter practical caps.
- Behavior under recommended versus pathological sampling configurations.
- Performance on unrelated capabilities and safety evaluations.

Conditional accuracy among completed outputs is not itself a clean estimate of mathematical knowledge, because the edited and unedited models may complete different subsets of questions. But tracking it alongside completion rate can help distinguish broad competence from an execution bottleneck.

A robust design could also compare the weight edit against early-stop heuristics, repetition penalties, alternative sampling parameters, and independent answer extraction from partially generated traces. Those controls would establish whether direct weight editing offers benefits beyond a much cheaper decoding change.

The original research already provides compelling evidence that loop behavior changed. A wider control study would tell us how much of the mathematical score jump is unique to the particular weight-space intervention.

## 25. What are the right baselines for a language-model weight programmer?

It is easy to compare a model-guided edit with doing nothing. That is not enough, especially when the model is given high-quality descriptions of the environment or a significant feedback budget.

A stronger benchmark would compare several strategies using the same architecture and evaluation conditions.

**Random sparse edits.** Randomly select the same number of currently nonzero weights or the same number of editable columns. Match the update magnitudes to those of the proposed edit. This establishes how much improvement might come from generic perturbation rather than targeted reasoning.

**Magnitude-based and activation-based baselines.** Suppress unusually large or highly correlated pathways using simple statistical rules. Our toy benchmark shows that this baseline can entirely solve an engineered failure, so omitting it would exaggerate the apparent novelty of the semantic editor.

**Gradient-free optimizers.** Compare against random search, finite-difference methods, evolution strategies, and simultaneous perturbation techniques at matched forward-pass budgets. These methods do not need gradient access, and they deserve to compete in a claim about gradient-free optimization. [9, 10]

**Gradient-based upper bounds.** For scientific understanding, it is useful to know what a small amount of supervised or reinforcement learning would achieve if training were permitted. The point is not to accuse weight programming of breaking its own rules, but to measure whether its imposed constraints are buying efficiency, interpretability, or some other advantage.

**Hand-coded circuit baselines.** In environments where the rules are available, compile a classical algorithm or simple controller into the same model architecture. The original Breakout result is precisely the kind of case where this control is essential.

**Decoding-only repairs.** For looping language models, test alternative generation settings, stopping rules, and repeat suppression. The Qwen model card itself suggests that the loop behavior is sensitive to inference configuration. [4]

**Alternative language-model editors.** Use smaller, cheaper language models with the same structural hints and tool budget, and measure whether the frontier model contributes meaningful reasoning beyond the prompt and harness.

For every baseline, report not only the final task score but also the number of evaluations consumed, input information provided, tool calls, wall time, memory use, and the number of model parameters modified.

## 26. Designing a serious evaluation harness

The best part of Weight-Space Programming may eventually be the **harness**, not the spectacular individual examples.

The correct research design treats a model editor as a potentially fallible program that proposes changes to another program. It cannot be allowed to choose its own grader, rewrite the test set, or invent scores. Our lab adopts that separation on a miniature scale; a larger system should make it a security and reproducibility property.

### Step 1: Freeze the original checkpoint

Record a cryptographic hash of every tensor file and architecture configuration. Do not accept a candidate edit unless its reported diff is relative to the exact base checkpoint. An edit applied to a different revision can silently affect unrelated behavior.

### Step 2: Separate read access from edit access

During diagnosis, permit controlled inspection of weights and perhaps a declared set of intermediate observables. During modification, allow updates only to a named subset of tensors or columns. Any change outside the permission set should invalidate the attempt.

### Step 3: Give the editor a fixed evaluation budget

Count every forward-pass request, the number of examples within it, and the amount of returned diagnostic information. A single request containing hundreds of rollouts is not the same budget as one state evaluation.

### Step 4: Preserve every proposal

Store the candidate diff, the hypothesis motivating it, the requested evaluation, and the evaluator response. This makes it possible to reconstruct the sequence of reasoning and distinguish a first-shot success from a long search whose failures were omitted.

### Step 5: Run a hidden evaluator

The edit must be evaluated on data and environments unavailable to the editor during its search. Include distributions that distinguish the intended mechanism from shortcut behavior. For language models, preserve sampling settings and evaluation prompts across candidates, and evaluate several seeds or samples where stochasticity matters.

### Step 6: Measure edit locality and collateral damage

Locality requires both a numerical statement and a behavioral statement. A low fraction of changed parameters does not guarantee low semantic side effects. Test unrelated benchmarks, held-out behaviors, robustness, and, for safety-sensitive systems, known safety properties.

### Step 7: Include hard negative controls

Randomize irrelevant neuron indices, alter hidden representations without changing the function, inject known nuisance circuits, and compare against inexpensive statistical baselines. A method that fails these controls may be reading incidental parameter patterns instead of computation.

### Step 8: Publish raw traces, not only final charts

A credible replication package should include exact checkpoint identifiers, initialization methods, tool budgets, test cases or reconstructible manifests, all non-sensitive run traces, code revisions, and the scoring implementation. The MIT CSAIL team's released logs are a meaningful step in this direction, even though they do not include every development trial or every large intermediate artifact. [2]

## 27. The broader implications, from model repair to model security

The practical appeal of direct parameter editing is easy to understand. Training a model again can be expensive. Low-rank adapters and fine-tuning require data, optimization, and careful evaluations. A targeted edit, when the failure has a compact causal explanation, could be far more efficient.

Consider an enterprise model that produces a repeated XML fragment on one class of inputs, a robotic controller that stalls in a narrow state region, or a speech model that misroutes a particular acoustic event. If a mechanistic explanation identifies a small malfunctioning circuit, a narrowly scoped edit might resolve it more efficiently than additional full-model training.

But the same capabilities create new security and governance problems.

If a third party can modify a model's weights, a tiny checkpoint diff may embed a backdoor, change refusal behavior, selectively alter responses, or affect a narrow high-value task without shifting most benchmark metrics. Small parameter footprint is not a substitute for auditability.

Every deployment that permits weight editing should therefore require artifact provenance, approved edit scopes, immutable checkpoint backups, independent behavioral regression testing, and a rollback path. A model's own explanation of why its edit is safe is not sufficient evidence.

There is also an access-control question. Research on weight or activation interventions is useful for interpretability, but unrestricted modification of safety-critical model pathways can be misused. The responsible lesson is to build measured, reversible, inspectable interventions rather than treat every editable direction as a feature toggle.

## 28. What the work has not established

It would be premature to conclude that general-purpose LLMs can now optimize arbitrary neural models more efficiently than gradient descent. The published experiments are early, carefully chosen, and heterogeneous. Some tasks are small enough for direct inspection. Others provide source code, task rules, one failure trace, Python analysis, and limited feedback opportunities. [1]

Several limits remain open.

First, a language model's success on a handful of networks does not imply robustness across random initializations, weight permutations, rescalings, quantized checkpoints, or unseen architecture families.

Second, neural algorithms compiled from known task rules should be compared against traditional rule-based algorithms, not only against policies that must infer those rules through optimization.

Third, the Qwen intervention addresses an especially severe execution pathology. Its failure mode is at least partly known from the model card, and the Gemma follow-up shows that a comparable improvement is not automatic when ordinary reasoning limitations dominate.

Fourth, reasoning token usage and the human effort required to design the prompt and harness should be counted as part of the optimization cost. A supposed zero-training method can still require substantial inference, search, debugging, and evaluation compute.

Fifth, the reported edited checkpoint and logged results support inspection, but the full research methodology still needs larger independent replications that run the models under new seeds and broader test distributions.

Finally, a clever weight edit is a causal intervention, not necessarily a complete mechanistic explanation. It shows that changing some pathway affects behavior. It does not prove that the pathway exclusively represents the target concept, nor that the same intervention works in every context.

These limits do not diminish the result. They tell us what the next rigorous experiments should measure.

## 29. Run the complete companion project

Extract the ZIP archive accompanying this article. Use Python 3.10 or newer and create a fresh virtual environment if desired.

```bash
cd weight_space_programming_lab

python -m pip install -r requirements.txt

# Verify algebraic and implementation properties
python -m unittest discover -s tests -v

# Inspect hand-compiled circuits and a repair
python -m weightlab.demo

# Reproduce all five random seeds and seven edit policies
python -m weightlab.benchmark \
  --seeds 5 \
  --readings 60 \
  --samples 3000 \
  --episodes 300 \
  --output results/benchmark.json

# Generate the figures from saved results
python plot_results.py
```

All core experiments are CPU-friendly. The repository does not require a network connection at runtime, and no API token or pretrained model download is necessary.

For a checkpoint-level audit, install the optional `safetensors` dependency and compare two trusted local single-file checkpoints:

```bash
python -m pip install safetensors
python audit_safetensors.py original.safetensors edited.safetensors
```

This command reports the number of differing tensor elements and affected tensors. It does not run a checkpoint, reproduce a benchmark score, or prove that an edit is safe. Sharded checkpoints require handling each shard and checking the corresponding manifest.

The original MIT experiment can be investigated separately through the [released logs](https://github.com/RikiyaT/lm_understand-logs). Those logs contain the original prompts and evaluation artifacts, not a turnkey replacement for our independent CPU demonstration.

### How to extend the lab into a real research benchmark

A substantial next project would start with networks actually trained on several datasets, rather than our hand-compiled circuits. Hide task labels and input meanings from the reader. Train multiple seeds, then generate permutation and scaling variants that preserve functionality.

Invite candidate editors, including LLMs and classical algorithms, to examine the checkpoints under a strict diagnostic budget. Give them access to the same number of probes. Collect proposed edits as signed tensor diffs. Run those diffs against hidden tasks, unseen seeds, and a catalog of negative controls.

The final score should jointly measure task improvement, unrelated task retention, changed-parameter footprint, forward-pass budget, time, and reproducibility. Only then can a claim about an LLM being a better weight programmer than an optimizer be evaluated fairly.

## 30. What is the actual shift here?

The story is not that gradient descent has suddenly become obsolete. It is that a new interface to model development is becoming plausible.

Instead of treating neural weights as an opaque artifact produced by training, an engineer may increasingly treat them as a partially readable and editable program. In the best cases, an agent can identify a specific mechanism, implement a minimal intervention, and explain which observations should change if its hypothesis is correct.

The original research shows tantalizing examples of all three steps: interpretation, construction, and repair. The failed sycophancy test and the unsuccessful Gemma math edit are just as important as the headline successes. They demonstrate that weight understanding is still incomplete and that removing a failure is easier than conjuring a missing ability.

Our own toy experiment reinforces a different lesson. We can write a neural function by hand. We can repair a very specific defect by editing two parameters. We can also find that a simple top-magnitude pruning rule solves exactly the same problem. And we can prove that a weight interpretation collapses when an arbitrary hidden-neuron ordering changes, even while the network's behavior remains identical.

These are small results, but their methodological implications are large.

**The future of Weight-Space Programming will not be decided by how few weights an AI can change. It will be decided by whether it can explain why those weights matter, prove that the edit is causal, and outperform the simplest alternative without breaking anything else.**

---

## 31. References and further reading

All percentages attributed to the original MIT CSAIL study refer to the authors' reported experiments, unless explicitly identified as an independent audit or our synthetic benchmark.

**[1]** Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab. *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL research blog, November 2, 2026. Primary experimental description, protocols, Qwen results, MinAtar comparisons, and negative findings. https://rikiyat.github.io/blog/weight-space-programming/

**[2]** RikiyaT. *lm_understand-logs.* Public repository of experiment prompts, model outputs, trace data, edits, final networks, and evaluations. See the individual experiment READMEs and license file. https://github.com/RikiyaT/lm_understand-logs

**[3]** Nelson Elhage et al. *A Mathematical Framework for Transformer Circuits.* Transformer Circuits, 2021. Background for circuit composition and induction behavior. https://transformer-circuits.pub/2021/framework/index.html

**[4]** Qwen Team. *Qwen3.5-0.8B model card.* Documents recommended inference settings and the risk of extended thinking loops in the 0.8B model. https://huggingface.co/Qwen/Qwen3.5-0.8B

**[5]** Carlos Toledo. *Weight-Space Programming: the edit decided from its bytes, the scores re-graded.* November 2026. A separate, technically scoped audit of released checkpoint differences and graded outputs. It is not a fresh end-to-end model evaluation. https://carlostoledo.co/reports/weight-space-programming.html

**[6]** Kevin Meng et al. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. Rank-One Model Editing and causal localization. https://arxiv.org/abs/2202.05262

**[7]** Andy Arditi et al. *Refusal in Language Models Is Mediated by a Single Direction.* 2024. An influential example of low-dimensional behavior intervention. https://arxiv.org/abs/2406.11717

**[8]** Faaiz Joad et al. *There Is More to Refusal in Large Language Models than a Single Direction.* 2026. A more nuanced account of refusal-related directions and shared behavioral tradeoffs. https://arxiv.org/abs/2602.02132

**[9]** James C. Spall. *Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation.* IEEE Transactions on Automatic Control, 1992. A classical evaluation-based optimization baseline. https://doi.org/10.1109/9.119632

**[10]** Tim Salimans et al. *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.* 2017. Gradient-free policy search with environment feedback. https://arxiv.org/abs/1703.03864

**[11]** Rikiya Takehi et al. *Experiment 8: Qwen math editing.* Original run records, checkpoint diffs, and evaluation artifacts, inside the released research repository. https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math

**[12]** The companion project for this article. *Weight-Space Programming: a controlled PyTorch lab.* Independent, deliberately constructed examples, reproducible seeds, tests, full JSON results, and saved figures. This project does not reproduce the original authors' Astra calls or Qwen benchmark.

---

*If you work with neural networks, the next valuable question is no longer merely whether your model can be edited. It is whether the proposed edit has a verified causal explanation, a measured side-effect budget, and a baseline strong enough to make the result meaningful.*
