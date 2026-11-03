# The Transformer Could Learn 17 Games. A Small Working Memory Helped It Remember What to Do Next.

## A deep technical tutorial on Decision Transformers with Working Memory, content-addressed read and write operations, task adaptation, and the real limits of memory-augmented reinforcement learning

A robot learns to move an object across a table. Then it learns to push a button. Then it learns to open a drawer.

Every new task teaches it something about the world. But if all of that knowledge must be compressed into the same fixed set of neural network weights, learning something new can interfere with what it already knows. And if the robot encounters an unfamiliar task, it has no obvious place to put the few things it learns during its first attempts.

We often describe this as a problem of model capacity. Give the model more parameters. Give it more trajectories. Train it longer.

But there is another possibility.

**What if the model does not need to learn a different weight configuration for every situation? What if it needs somewhere to remember the useful parts of the situation it is currently in?**

That is the premise of *Think Before You Act: Decision Transformers with Working Memory*, a paper by Jikun Kang, Romain Laroche, Xingdi Yuan, Adam Trischler, Xue Liu, and Jie Fu, published at ICML 2024 [1]. The authors add a compact, content-addressed memory matrix to a Decision Transformer, allowing it to write information associated with experiences and retrieve relevant latent representations before producing actions.

The result is a model they call **DT-Mem**. In the authors' reported experiments, it improves performance on held-out Atari games and Meta-World manipulation tasks while requiring less training time than several much larger Decision Transformer baselines [1, 2]. The paper also explores adapting the memory's projection weights with LoRA instead of changing the entire decision-making backbone.

Those results are interesting. The mechanism behind them is even more useful to understand.

In this tutorial, we will build the idea from first principles, inspect what the original research does and does not establish, and implement a small working-memory lab in PyTorch. We will benchmark support-data availability, memory capacity, corrupted writes, and broken context-action associations. Finally, we will build a tiny causal Decision Transformer and a separate LoRA exercise, then discuss what would be required to reproduce the original Atari and robotics results.

**A boundary worth setting immediately:** the companion experiments are independent teaching implementations. They do not reproduce the original DT-Mem model, its learned memory gates, or its Atari and Meta-World benchmarks. That distinction will stay visible throughout the article.

---

## Contents

1. Why a decision-making model needs working memory
2. The original Decision Transformer in five minutes
3. Offline reinforcement learning is not online learning
4. What DT-Mem adds to the architecture
5. Memory as a differentiable matrix
6. Step 1: Encode a trajectory
7. Step 2: Find a memory address
8. Step 3: Decide what to erase and write
9. Step 4: Retrieve memory before acting
10. How to understand the complete data flow
11. Memory is not a database or a larger context window
12. Why LoRA is used for adaptation
13. What the authors actually measured
14. How to read the Atari results responsibly
15. The compute comparison and its caveats
16. The Meta-World and LoRA results
17. Build the educational PyTorch lab
18. Reproduce a causal Decision Transformer
19. Implement the attention-addressed memory
20. Design a task that reveals the value of memory
21. Establish a baseline without task-specific examples
22. Run our five-seed experiment
23. Ablation I: How many examples do we remember?
24. Ablation II: How much memory is enough?
25. Ablation III: What happens when memory is wrong?
26. A small LoRA experiment with frozen base weights
27. Benchmark methodology that would withstand scrutiny
28. Failure modes and limitations
29. What this research suggests for today's agents
30. A roadmap for a faithful reproduction
31. Conclusion and references

---

## 1. Why a decision-making model needs working memory

Imagine two versions of an agent. Both know how to recognize an object, understand a goal, and predict an action.

The first agent has to express everything it knows through its parameters and the limited history that fits inside its context window. When it moves between tasks, it relies on whatever patterns those parameters have encoded.

The second agent has the same model, but also carries a small stateful workspace. When an experience matters, it writes a representation into that workspace. When a new decision resembles an earlier experience, it reads the corresponding representation back.

The second agent has an additional resource: **task-local information that can change without changing the backbone model weights**.

That distinction matters because three different problems are commonly grouped together under the word *memory*.

**Parametric memory** is information stored by training in weights. It is persistent, but adapting it normally requires optimization. Updates can interfere with previously learned behavior.

**Sequence context** is information supplied in recent tokens. It is accessible through attention, but it has a computational cost and a finite window. Its structure is usually determined by the input format rather than an explicit memory allocation policy.

**Working memory** is a set of representations the model can actively address, update, and reuse while solving tasks. It may be shorter lived than long-term weights, but it can change quickly and condition the policy on task-specific experience.

The DT-Mem paper is motivated by catastrophic forgetting and inefficient multi-task learning. Its central bet is that a learned mechanism for deciding where to place information and what to retrieve can improve policy generalization and adaptation. This is an architectural hypothesis, not proof that all forgetting originates in missing working memory [1].

The inspiration comes from a long line of memory-augmented models, including Neural Turing Machines, memory networks, and other content-addressable architectures [3, 4].

## 2. The original Decision Transformer in five minutes

A conventional reinforcement learning algorithm often trains a policy with a value function, a Bellman target, or a policy-gradient objective. The Decision Transformer takes a different route.

Chen and colleagues asked whether we could learn to act by modeling good trajectories as sequences [5].

A trajectory contains states, actions, and rewards. The Decision Transformer reorganizes those elements into a sequence conditioned on a desired future return. Instead of estimating a value function and improving a policy through a separate control update, it learns to predict actions from a recent history of desired returns, observations, and earlier actions.

Conceptually, the token ordering looks like this:

```text
Desired return at t0 | State at t0 | Action at t0
Desired return at t1 | State at t1 | Action at t1
Desired return at t2 | State at t2 | Action at t2
...
```

A causal Transformer processes those tokens. When predicting the action for time `t`, it may see the desired return, the current state, and the previous history. It must **not** see the action it is trying to predict.

The return-to-go is the sum of rewards remaining from a given point in a trajectory. It acts as an instruction for the behavior the model should reproduce. In an offline dataset, actions associated with high desired returns can help the model learn successful behavior.

The crucial reframing is that **policy learning becomes conditional sequence modeling**. It makes Transformer tools available to offline control without requiring a standard Bellman-style training loop [5].

That does not magically solve exploration, out-of-distribution actions, or poorly covered datasets. A Decision Transformer can still only learn reliable behavior from the information its training data supports [6].

## 3. Offline reinforcement learning is not online learning

Before we discuss memory, we need to be precise about the experimental setting.

In **offline reinforcement learning**, the training data consists of trajectories collected beforehand. The agent does not continuously query the environment for new experiences during pretraining. It tries to infer a good decision policy from the dataset it is given.

This is important because the ability to generalize to an unseen game is not the same as the ability to adapt after seeing a few examples from that game.

Kang and colleagues explicitly distinguish **generalization** from **adaptability**. Generalization refers to performance on held-out tasks without additional model fine-tuning. Adaptability refers to improvements obtained after fine-tuning on limited data from the new task [1].

In their Atari evaluation, the authors also examine held-out games after fine-tuning. When reading the paper's results, look for the specific evaluation protocol rather than treating every held-out score as a strict zero-shot result. The dataset includes expert and non-expert DQN trajectories, and successful return-conditioned prediction depends on the quality and coverage of those trajectories [1, 7].

The memory module addresses **how relevant information is represented and reused**. It does not automatically create missing data.

## 4. What DT-Mem adds to the architecture

The original DT-Mem architecture has four main pieces [1]:

1. A trajectory encoder that embeds states, actions, and return-to-go values.
2. Attention blocks that model relationships across the trajectory.
3. A working-memory module that updates and retrieves latent information.
4. An action decoder that converts the resulting representation into an action prediction.

A simplified flow looks like this:

```text
Offline trajectories
    |
    v
Return-to-go + state + action embeddings
    |
    v
Transformer attention blocks
    |
    v
Latent sequence representation
    |
    +-------------------------------+
    |                               |
    v                               v
Memory addressing              Current memory
    |                               |
    +--------- write/update --------+
                    |
                    v
             Updated memory
                    |
                    v
              Memory read
                    |
                    v
              Action decoder
                    |
                    v
                Next action
```

The central innovation is not merely concatenating a stored vector to the prompt. The memory is designed to receive information from the Transformer, assign content-based addresses, change selected storage locations, and provide an updated representation for the action decoder.

The paper reports a Transformer component of roughly 13 million parameters, about 7 million memory-module parameters, and a primary DT-Mem model of about 20 million parameters in total [1]. Its architecture table also lists a memory-size figure of 559K. But the appendix gives 1,290 slots with 512 dimensions, which would contain 660,480 scalar values if interpreted as one simple slot matrix. Those counts do not reconcile directly in the paper text. A careful reproduction should verify the precise buffer shape in the implementation rather than silently treating all three numbers as interchangeable.

Notice the distinction between the **size of the memory state** and the **number of trainable memory-module parameters**. A memory matrix is mutable state; the query, key, value and other learned projections constitute trainable weights. Treating these quantities as interchangeable produces misleading comparisons.

## 5. Memory as a differentiable matrix

A conventional key-value store uses explicit keys. Ask for `user_id=17`, and the system retrieves a corresponding record.

Neural working memory instead stores vectors. The address can be computed from similarity between a current representation and the representations already associated with memory slots.

Suppose the memory contains `N` slots. Each slot is a vector of dimension `d`. Its conceptual shape is:

```python
memory.shape == (num_slots, hidden_dim)
```

For example, 8 memory slots with 16 dimensions each can be represented by a PyTorch tensor of shape `(8, 16)`.

The system needs to solve three questions at each step.

- Which slot is relevant to what the agent just observed?
- How much should that slot change?
- Which stored information should contribute to the next action?

The DT-Mem paper answers these questions with attention-derived addressing weights, erase/add updates, and content-based retrieval [1].

An important detail: **the paper starts its memory from a random matrix**, whereas our teaching code will use fixed context prototypes for transparent addressing. Our lab is designed to isolate the effect of receiving and preserving relevant examples, not to reproduce the training of neural memory addresses.

## 6. Step 1: Encode a trajectory

The input to a Decision Transformer is not just an observation. It is a structured history.

DT-Mem embeds states, actions, and desired returns in a common latent space before passing them through attention. In the original work, the context length is 28 timesteps, and the hidden dimension is 512 in the reported primary model [1].

For an implementer, the crucial question is whether the model can reconstruct a useful action prediction from its permitted prefix.

Consider an episode with the following data:

```text
Time 0: return-to-go 3, state A, action LEFT
Time 1: return-to-go 2, state B, action UP
Time 2: return-to-go 1, state C, action RIGHT
```

At time 2, the model can see the earlier actions. It can also see the current desired return and state. But if the training implementation allows the time-2 action token to influence the prediction of the time-2 action, it has leaked the answer.

This mistake is easy to make when writing a custom Transformer. The model may achieve excellent training accuracy while learning a meaningless identity mapping from current action input to current action target.

We will build a causal masking test later to catch exactly this bug.

## 7. Step 2: Find a memory address

DT-Mem produces query and key projections from the memory and sequence embeddings. The paper describes computing a softmax over scaled query-key dot products [1].

The intuition is straightforward. When an incoming experience is similar to a memory entry, that entry should receive a larger address weight.

Here is the general pattern using PyTorch:

```python
import torch

# queries: [batch, hidden_dim]
# keys:    [slots, hidden_dim]

scores = queries @ keys.T
weights = torch.softmax(scores / (queries.shape[-1] ** 0.5), dim=-1)

# weights: [batch, slots]
```

Each row in `weights` sums to one. A query can focus heavily on a single slot, or spread mass across several slots.

Two implementation details matter.

**Attention weights are not inherently reliable addresses.** The model must learn representations in which similarity correlates with task-relevant information. If two unrelated situations look similar in the learned latent space, their memory can interfere.

**Softmax does not create more capacity.** If the model has very few slots, many experiences may compete for the same storage locations. The resulting mixture can lose the information the model was meant to preserve.

Our ablations will deliberately reduce slot capacity to expose this limitation.

## 8. Step 3: Decide what to erase and write

Finding an address is only half the problem. A working memory that can only append new values will eventually become cluttered. A working memory that always overwrites its most relevant slot may destroy something useful.

DT-Mem uses a second attention-derived writing strength and constructs erase and add terms [1]. In compact pseudocode, the intended pattern is:

```python
address = attention(memory, current_experience)
write_strength = attention(memory, current_experience)

erase = address * (1 - write_strength)
add = combine(address, write_strength, current_experience)

updated_memory = old_memory * (1 - erase) + add
```

This pseudocode captures the **conceptual sequence**, not the exact shape conventions or every projection tensor in the authors' implementation.

The erase term reduces selected existing values. The add term injects the new information. The result can blend experiences rather than force a full replacement.

Why can blending help? Imagine a robot repeatedly seeing that a particular grasp position succeeds for a class of objects. A distributed update could strengthen the representation for that class while retaining useful prior information.

Why can blending hurt? If the learned address is wrong, the same operation can store conflicting evidence in one location. A model may gradually convert a useful memory into an average of incompatible policies.

This tradeoff is not incidental. It is the heart of the architecture.

## 9. Step 4: Retrieve memory before acting

After updating the matrix, DT-Mem reads the memory using content-based addressing. The retrieved representation contributes to action prediction [1].

A generic memory read looks like:

```python
# weights: [batch, slots]
# memory:  [slots, hidden_dim]

retrieved = weights @ memory

# retrieved: [batch, hidden_dim]
```

This weighted sum is a standard, shape-explicit illustration of content-addressed retrieval. The paper's notation expresses retrieval as an elementwise weighting of memory; exact aggregation and downstream processing depend on the implementation.

A useful engineering insight follows: **the quality of the stored representations and the quality of the address mechanism are distinct concerns**.

A model might store the right fact but retrieve the wrong slot. Or it might retrieve the right slot after the contents were corrupted by earlier writes. Either failure can produce an incorrect action, and the final action alone does not identify which component was responsible.

That is why a proper memory benchmark needs ablations for addressing, capacity, and write quality instead of only comparing a final task score.

## 10. How to understand the complete data flow

Think of one action prediction as three interacting processes.

The **Transformer** recognizes patterns in the recent trajectory and produces a latent representation of what is happening now.

The **memory controller** interprets that representation in relation to stored task knowledge. It decides where to write and what to retrieve.

The **action decoder** combines the representation coming back from memory with the learned policy structure to predict the next action.

When this works, a previously seen situation can help the next decision even without a large parameter update. When it fails, the agent might preserve the wrong information, mix distinct tasks, or fail to access the memory it needs.

The original model optimizes these components with offline supervised objectives that include action predictions and additional reward-related prediction losses [1]. These auxiliary targets matter because they shape the representation the memory controller will encounter during training. Simply attaching a memory matrix to an arbitrary pretrained Transformer is not guaranteed to reproduce the original results.

## 11. Memory is not a database or a larger context window

It is tempting to compare DT-Mem directly with retrieval-augmented generation. Both systems retrieve relevant information. But their interfaces are different.

A retrieval system typically searches an external collection of documents or trajectory fragments, often with a nearest-neighbor method. Its storage may be persistent and arbitrarily large. The retrieved content can remain human-readable or be converted into model embeddings.

DT-Mem's working memory is a compact latent matrix integrated into the model's forward computation. Reading and writing are part of the model's internal computation, not necessarily a separate call to a search index [1].

Likewise, a longer Transformer context provides access to more previous tokens, but it does not by itself learn which past information deserves a persistent, editable slot. Ordinary self-attention and dedicated memory solve related but not identical problems.

There is an important tradeoff. External retrieval is easier to inspect and refresh. Latent memory can be compact and tightly coupled to the policy, but its stored content is harder to interpret. In safety-critical robotics, this interpretability cost matters.

## 12. Why LoRA is used for adaptation

Pretraining a decision model across many tasks can be expensive. Fine-tuning the entire model on a small new dataset also changes a large number of parameters and may disturb behaviors learned during pretraining.

LoRA, or Low-Rank Adaptation, addresses part of the optimization cost [8]. Instead of updating the entire weight matrix, it freezes the original weights and learns a low-rank correction.

Consider a frozen weight matrix `W`. LoRA adds `B @ A`, where the rank `r` is much smaller than either matrix dimension.

```text
Frozen projection:         y = W x
Adapted projection:        y = (W + B A) x
Trainable parameters:      B and A
Frozen parameters:         W
```

The DT-Mem paper applies low-rank updates to learned memory projection weights. The backbone remains fixed for downstream fine-tuning, and the adapter modifies how information is addressed and processed within memory [1].

That is different from loading more examples into the memory matrix at inference time. **Updating memory contents and fine-tuning the parameters of the memory controller are two different adaptation mechanisms.**

The former changes state. The latter changes how the system reads and writes that state.

We will demonstrate the mathematics of LoRA with a separate minimal code example. Our lab does not reproduce the paper's Meta-World adapter training.

## 13. What the authors actually measured

Research claims are most useful when separated from the experimental conditions that generated them.

The paper evaluates DT-Mem on two families of environments [1].

For **Atari**, the authors work with a multi-game offline dataset built from DQN trajectories. They select 17 training games from the larger collection and use held-out games to evaluate generalization and adaptation. They compare against multi-game and recurrent-memory Decision Transformer baselines.

For **Meta-World**, they use the ML45 benchmark, with 45 training tasks and five testing tasks. They compare with prompt-conditioned and hypernetwork-based alternatives.

The reported primary model uses four Transformer layers, hidden dimension 512, eight attention heads, 1,290 memory slots, and LoRA for downstream memory adaptation. Its total model size is approximately 20 million parameters. The authors report training on eight NVIDIA V100 GPUs and fine-tuning on one V100 [1].

These details matter if you intend to compare the system fairly with a modern model. The number of environment steps, quality of the offline demonstrations, training compute, model size, and fine-tuning budget all affect the outcome.

## 14. How to read the Atari results responsibly

The authors report DQN-normalized scores on four held-out Atari games, with uncertainty estimates across repeated evaluations [1].

For *Alien*, the listed scores are 3.8% for MDT, 22.3% for the recurrent-memory baseline, and 51.0% for DT-Mem.

For *Ms. Pac-Man*, the corresponding results are 13.2%, 22.9%, and 69.3%.

For *Space Invaders*, they report 8.6%, 17.6%, and 53.6%.

For *Star Gunner*, they report 2.3%, 27.7%, and 62.2%.

These are **reported research results**, not outcomes from our synthetic Python lab. The paper labels the values as DQN-normalized scores and provides confidence intervals, rather than reporting them as universal percentages of game completion. The uncertainty is substantial for some titles, notably Alien [1].

The result suggests that an addressable memory can be beneficial in the authors' experimental regime. It does not establish that adding memory to any Transformer will produce equivalent gains, or that the system has solved continuous learning and forgetting in a general setting.

Another subtlety is that the paper discusses both training-time generalization and later fine-tuning performance. If a score is produced after adapting on data from the held-out game, it should not be described as pure zero-shot task success.

## 15. The compute comparison and its caveats

The paper lists the following **training-time measurements** [1]:

- DT-Mem-20M: 50 hours.
- MDT-13M: 200 hours.
- MDT-40M: 400 hours.
- MDT-200M: 1,600 hours.

These are training times in hours, as labeled in the paper. They must not be casually relabeled as GPU-hours. Since the paper says the models were trained using eight V100 GPUs, wall-clock hours and aggregate accelerator-hours are different quantities.

The relative times are striking. DT-Mem is four times faster than the smallest listed MDT reference, eight times faster than MDT-40M, and 32 times faster than MDT-200M in the reported experiment.

But **different model sizes are not a single controlled architectural ablation**. For a strict efficiency comparison, we would want equivalent quality targets, matched hardware, the same data processing pipeline, the same amount of tuning, measured throughput, and consistent stopping criteria. A comparison between 20 million and 200 million parameters includes architectural and scale differences.

A careful interpretation is that the paper provides promising evidence for a favorable performance-compute tradeoff in its evaluated setup, not a universal 32-fold speed advantage for working-memory Transformers.

## 16. The Meta-World and LoRA results

The Meta-World results are interesting because they isolate another question: can we adapt a pretrained decision-making system to a new task while tuning relatively few parameters?

The paper reports a testing score of 0.20 for DT-Mem before fine-tuning and 0.95 afterward. It reports 0.12 and 0.92 for the hypernetwork baseline, and 0.06 and 0.09 for the prompt-based baseline [1].

It also reports 147,000 adaptation parameters for DT-Mem in the configuration described, compared with a hypernetwork-based approach with additional pretrained network overhead. The LoRA parameter count is approximately 0.7% of the primary model size in the paper's accounting.

Those numbers support the authors' claim that memory-controller adaptation can be parameter-efficient in this benchmark. They do not mean that fine-tuning is free or that the system learns an entirely new manipulation skill from a single example.

In fact, the authors explicitly acknowledge that the fine-tuning dataset remains substantial. They discuss using 10% of the held-out Atari data, with 50,000 environment steps in the described Atari setup, and 10,000 episodes for the Meta-World fine-tuning protocol [1].

The gap between *parameter efficiency* and *data efficiency* is an important one. A small adapter can still need considerable experience to learn a new task.

---

# Part II. Rebuild the ideas in PyTorch

## 17. Build the educational PyTorch lab

The companion project is intended to run on a laptop CPU. No Atari ROMs, robot simulator, GPU, external service, or API key is required.

It contains five independently understandable components:

- A tiny causal Decision Transformer that processes return, state, and action tokens.
- A softmax-addressed working-memory module with explicit storage slots.
- A synthetic task distribution where each held-out task has a new action mapping.
- A PyTorch baseline trained on many task mappings but given no information identifying a new task.
- A LoRA example that changes a frozen projection with low-rank factors.

Our **working-memory benchmark uses the second, third, and fourth components**. The tiny Transformer and LoRA implementations are separate architectural exercises. We do not falsely combine them into a claimed reproduction of DT-Mem.

After downloading and extracting the lab, install PyTorch:

```bash
cd dt_mem_working_memory_lab

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m unittest discover -s tests -v
```

On Windows PowerShell, activate the virtual environment using `.venv\Scripts\Activate.ps1`.

The code was tested with PyTorch 2.10.0 on CPU. It is designed for Python 3.10 or later.

## 18. Reproduce a causal Decision Transformer

Start with a tiny Transformer. This is an architectural exercise in action-conditioned sequence modeling, not the original DT-Mem implementation.

```python
import torch
from torch import nn

class TinyDecisionTransformer(nn.Module):
    def __init__(self, n_states=8, n_actions=4, width=32, heads=4):
        super().__init__()
        self.state = nn.Embedding(n_states, width)
        self.action = nn.Embedding(n_actions + 1, width)
        self.ret = nn.Linear(1, width)
        self.pos = nn.Embedding(512, width)

        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=1)
        self.head = nn.Linear(width, n_actions)

    def forward(self, returns, states, actions):
        batch, time = states.shape
        r = self.ret(returns.float().unsqueeze(-1))
        s = self.state(states)
        a = self.action(actions)
        tokens = torch.stack([r, s, a], dim=2)
        tokens = tokens.reshape(batch, 3 * time, -1)
        tokens += self.pos(torch.arange(3 * time))[None]

        forbidden = torch.triu(
            torch.ones(3 * time, 3 * time, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(tokens, mask=forbidden)
        return self.head(hidden[:, 1::3, :])
```

The companion file `dtmem_lab/tiny_dt.py` contains the production version used by our tests, including device-aware position indexing and input validation.

Look carefully at the last line. We take the Transformer output from **state token positions**, not action token positions. For time step `t`, the corresponding action token appears afterward in the causal sequence. The model therefore cannot read the very action it has been asked to predict.

A unit test changes the current and future action tokens and confirms that earlier action predictions do not change.

That property is much more important than making the architecture look visually similar to a paper diagram. Without correct causal masking, the benchmark can accidentally reward a model for cheating.

The project includes a separate trainer that makes this tiny model actually learn a fixed synthetic action mapping:

```bash
python -m dtmem_lab.dt_train --steps 160
```

In the reference CPU run, this 25,540-parameter Transformer reached **100% action accuracy on fresh sequences from the same fixed task mapping** after 160 optimization steps. This is a deliberately easy behavior-cloning sanity check, not cross-task generalization or a memory benefit. It confirms that the causal token path can learn an action rule, while the later memory experiments isolate task-specific information access.

## 19. Implement the attention-addressed memory

We will now build a memory that can answer a simpler question: if a task reveals that context 3 requires action 2, can the model use that experience later without updating its weights?

Our teaching memory is a matrix of action evidence. Each row corresponds to a content address. A softmax chooses how strongly a context maps to each row. The values are accumulated, normalized, and read back.

Here is the central addressing operation from `dtmem_lab/memory.py`:

```python
def addressing(self, contexts):
    contexts = torch.as_tensor(contexts, dtype=torch.long).reshape(-1)
    x = torch.nn.functional.one_hot(
        contexts, num_classes=self.num_contexts
    ).float()
    return torch.softmax(
        self.temperature * (x @ self.keys.T), dim=-1
    )
```

The `keys` tensor represents fixed context prototypes. The temperature controls how sharply the address distribution focuses on the best matching slot.

A write stores action evidence in the corresponding slots:

```python
for context, action in zip(contexts, actions):
    weight = self.addressing(context).squeeze(0)
    action_value = torch.nn.functional.one_hot(
        action, num_classes=self.num_actions
    ).float()
    self.values += weight[:, None] * action_value
    self.mass += weight
```

A read first normalizes stored evidence, then combines it according to the query's attention weights:

```python
weights = self.addressing(query_contexts)
slot_actions = self.values / self.mass[:, None].clamp_min(1e-8)
probabilities = weights @ slot_actions
```

This is a **simplified teaching mechanism**. It has no jointly learned query/key/value projections and does not implement the paper's exact learned erase/add rule. It also does not decide autonomously which information is salient. We explicitly provide context-action observations.

Those simplifications are deliberate. They let us isolate and measure the informational advantage of memory before tackling the harder problem of learning representations that make addressing reliable.

## 20. Design a task that reveals the value of memory

We need an experiment in which task-specific memory matters and where the reason is easy to diagnose.

Construct eight distinct contexts and four possible actions. Every task maps the contexts to actions using a fresh random but balanced mapping. Each action is correct for exactly two contexts.

For example, one task might behave as follows:

```text
Context 0 -> Action 2
Context 1 -> Action 0
Context 2 -> Action 3
Context 3 -> Action 1
Context 4 -> Action 0
Context 5 -> Action 3
Context 6 -> Action 2
Context 7 -> Action 1
```

The next task generates another random mapping.

The baseline has seen many tasks, but the mapping for the new task is statistically independent of the ones in its training data. If we do not tell it anything about the new task, **there is no recoverable rule that lets it infer the correct mapping from the context alone**. With four balanced actions, expected chance accuracy is 25%.

This is a contextual decision-making toy environment. It is not a full Markov decision process with long-horizon reward, exploration, or delayed credit assignment. We are not pretending otherwise.

The point is to create a clean distinction between a model that only has general training experience and a model that also has a few **current-task examples**.

If the memory receives examples from the new task and retrieves the correct association at query time, it should improve. If its stored associations are corrupted or the capacity is too small, performance should fall.

## 21. Establish a baseline without task-specific examples

Our baseline is a small neural classifier:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(8, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 4),
)
```

For each optimization step, we sample a new task mapping, generate random contexts, and train the model to predict the corresponding actions.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)

for step in range(180):
    task_mapping = sample_task(generator)
    contexts = torch.randint(0, 8, (64,), generator=generator)
    inputs = torch.nn.functional.one_hot(contexts, 8).float()
    targets = task_mapping[contexts]

    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The complete implementation is in `dtmem_lab/toy_tasks.py`.

Because the mapping changes randomly between tasks, a context-only classifier cannot learn a reliable new-task assignment. This baseline acts as a check that task-local examples are genuinely additional information.

A subtle but essential caveat: this is **not** an equal-information comparison. The memory system is given current-task demonstrations; the baseline is not. The benchmark demonstrates the value of possessing and retrieving those demonstrations, not that our memory architecture is inherently better than every alternative.

A stronger research comparison would give the same demonstrations to a nearest-neighbor baseline, a small supervised adapter, and an in-context Transformer.

## 22. Run our five-seed experiment

The lab evaluates five independent random seeds. Each seed uses 100 held-out tasks, and each task contains 64 query decisions. That gives 32,000 query decisions across the full run for each configuration.

Every configuration sees the same task mapping and query contexts within a seed. The memory is reset between tasks, avoiding leakage of a previous task's mapping into the next.

Run:

```bash
python -m dtmem_lab.benchmark \
  --seeds 5 \
  --tasks 100 \
  --queries 64 \
  --steps 180 \
  --output results/benchmark.json
```

The saved reference run used **PyTorch 2.10.0+cpu**. The numbers below are the mean of five seed-level accuracies, with sample standard deviations across seeds.

The context-only baseline reaches **24.82% accuracy**, with a standard deviation of **1.01 percentage points**.

Adding eight memory slots and supplying 16 current-task examples raises accuracy to **91.74%**, with a standard deviation of **0.65 percentage points**.

That is a large difference, but it has a direct explanation. The memory knows associations the baseline has never been shown. For query contexts that appeared in clean support data, the full-capacity memory retrieves the correct action. For contexts missing from support, the system must fall back to the pretrained classifier.

Indeed, in our reference result, eight-slot memory with 16 clean examples achieves **100% accuracy on previously observed contexts**. That is partly a consequence of the benchmark construction: the address is an exact categorical identity, the action mapping is deterministic, and there is enough memory capacity for every context.

It is not evidence of a general agent learning new physical skills with perfect reliability.

## 23. Ablation I: How many examples do we remember?

Keep the memory capacity fixed at eight slots. Change only the number of task-specific examples it receives.

With **four examples**, accuracy is **55.52%**, with a standard deviation of **1.27 percentage points**.

With **16 examples**, accuracy is **91.74%**, with a standard deviation of **0.65 percentage points**.

With **32 examples**, accuracy reaches **99.12%**, with a standard deviation of **0.50 percentage points**.

This pattern is primarily a **coverage effect**. Four random observations are unlikely to visit every one of the eight contexts. Sixteen observations cover many more. Thirty-two usually cover nearly all of them.

The experiment is a reminder that memory and sample coverage cannot be separated. A model can have a perfectly functional memory yet fail on situations for which it has never stored useful evidence.

It also suggests a practical extension: rather than collecting task examples randomly, build a policy that asks for demonstrations from underrepresented or uncertain contexts. That would turn this toy experiment into a simple active data-collection problem.

However, selecting new experiences is precisely one of the limitations the DT-Mem paper identifies. Its primary setup is offline, and it does not propose a full online exploration strategy for acquiring the most informative new-task data [1].

## 24. Ablation II: How much memory is enough?

Next, keep 16 task-specific examples but reduce the number of slots.

With **eight slots**, accuracy is **91.74%**.

With **four slots**, accuracy falls to **57.32%**, with a standard deviation of **1.39 percentage points**.

With **two slots**, accuracy falls further to **37.87%**, with a standard deviation of **1.16 percentage points**.

Why? In the educational implementation, different contexts are deliberately assigned to shared slots when capacity is insufficient. The model then blends action evidence from multiple contexts. Even a correct memory read can return an ambiguous mixture because the information in the selected slot is no longer specific enough.

This is a controlled demonstration of **memory interference**.

The authors report a related capacity tradeoff in Appendix B of their paper. They observe improved performance with more slots up to a point, followed by declining performance when memory becomes too large relative to the available training budget [1]. This is not the same experiment: their memory keys and memory controller are learned, while our keys are fixed. But both highlight that more memory is not automatically better.

In a learned system, capacity affects parameter optimization, representation geometry, and the distribution of attention, not just how many examples can be stored.

## 25. Ablation III: What happens when memory is wrong?

Memory is useful only if the stored evidence is trustworthy.

We run two controlled corruption experiments.

First, we replace about 25% of the support labels with incorrect actions. The eight-slot, 16-example configuration drops from **91.74% to 73.55%**, with a standard deviation of **0.85 percentage points** for the corrupted condition.

Second, we randomly shuffle the support labels independently of their contexts. That breaks the relationship between a memory key and the action it should predict. Accuracy drops to **27.07%**, with a standard deviation of **2.27 percentage points**.

These are not small implementation details. They show that a system can perform a memory operation perfectly and still make a bad decision because the written content was wrong.

Imagine applying the same principle to a robot. If its experience logger accidentally associates a successful grasp with the wrong object state, content-based retrieval may confidently produce an unsafe action later.

This leads to three engineering requirements for real memory systems: validate important writes, track the source of stored information, and handle contradictory experiences rather than blindly accumulating them.

Our toy code does not solve those problems. It exposes why they matter.

## 26. A small LoRA experiment with frozen base weights

Now consider the other adaptation mechanism in the paper: **changing how memory representations are processed while leaving base parameters fixed**.

The companion lab includes a small `LoRALinear` module:

```python
class LoRALinear(torch.nn.Module):
    def __init__(self, frozen_weight, rank=2):
        super().__init__()
        out_features, in_features = frozen_weight.shape

        self.weight = torch.nn.Parameter(
            frozen_weight.clone(), requires_grad=False
        )
        self.A = torch.nn.Parameter(
            torch.randn(rank, in_features) * 0.02
        )
        self.B = torch.nn.Parameter(
            torch.zeros(out_features, rank)
        )

    def forward(self, x):
        return x @ self.weight.T + (x @ self.A.T) @ self.B.T
```

This excerpt sets the adapter scale to one for simplicity. The full lab supports an explicit `alpha / rank` scaling factor.

We create a synthetic target projection whose difference from the frozen weight matrix is exactly rank two. Then we train only the factors `A` and `B` to reconstruct that target.

Run:

```bash
python -m dtmem_lab.lora_demo
```

In the reference run, the initial mean squared error is about **0.304**. After optimization it falls to approximately **1.4e-14**. The frozen base projection contains 96 parameters, while the two trainable factors contain 40 parameters.

That result is deliberately easy because the target difference was constructed to have rank two. It verifies the mathematical idea of LoRA but says nothing about the ability of LoRA to adapt a real multi-task robot policy.

For DT-Mem, the interesting question is whether changing only the query, key, and value projections in the memory module is sufficient to recover task performance on unseen games. The authors report that it can be effective on their benchmarks [1]. A faithful experiment would need to train and evaluate those exact projections within the original model.

---

# Part III. How to evaluate this research properly

## 27. Benchmark methodology that would withstand scrutiny

A memory-augmented decision model is easy to make look impressive if its baseline is not allowed to observe the same evidence.

A serious comparison should separate **architectural contribution** from **information advantage**.

First, match the offline training data across models. This includes the distribution of successful and unsuccessful trajectories, normalization, frame preprocessing, and target-return conditioning.

Second, match the support data at evaluation time. If DT-Mem is given ten demonstrations from an unseen task, compare it with a nearest-neighbor policy, a supervised fine-tuned model, and a Transformer that receives those same demonstrations as context.

Third, control the optimization budget. Parameter count alone is not enough. Record accelerator-hours, wall-clock time, batch size, sequence length, tuning runs, and inference latency separately.

Fourth, split tasks rather than merely splitting observations. Randomly partitioning transitions from one task into train and test sets may create an easy interpolation problem instead of measuring adaptation to genuinely new dynamics.

Fifth, report uncertainty across independent training seeds. A result obtained from thousands of observations generated by one trained model does not automatically provide thousands of statistically independent replications.

Sixth, measure **returns through actual environment rollouts**, not just offline action-prediction accuracy. A classifier can imitate expert actions on familiar states while accumulating errors after one unfamiliar transition.

Seventh, include memory-specific stress tests. Vary slot count, support set quality, temporal delay, input noise, ambiguous observations, task switching frequency, and the probability of incorrect writes.

Eighth, test causal correctness. At decision time, no model input should contain future rewards, future states, or the target action itself. Our small Transformer includes a causal masking unit test for this reason.

Finally, pre-register which model selections depend on validation tasks. Repeatedly tuning against the same held-out tasks can silently turn the test set into another training set.

## 28. Failure modes and limitations

The DT-Mem authors acknowledge that memory fine-tuning is not yet very sample-efficient. They also note that they have not designed an online data-collection strategy and that a rigorous theoretical characterization of the architecture's advantages remains a future direction [1].

Beyond those stated limitations, implementation brings several additional questions.

**Interference between tasks.** Two tasks may map similar observations to opposite actions. A memory controller that keys only on visual similarity may confuse them unless its representation also preserves task identity or relevant context.

**Stale or misleading memories.** A learned policy might keep consulting an old memory after the environment changes. The system needs a strategy for overwriting, decay, conflict resolution, or versioning.

**Soft attention dilution.** A useful slot may receive only a small fraction of attention when many similar slots compete. More memory can make retrieval harder even if storage capacity increases.

**Expensive memory updates.** A compact memory state does not imply constant runtime. Attending over more slots, updating multi-dimensional representations, and preserving state through a long sequence all have costs.

**Unclear semantics.** A latent vector is not an auditable instruction. For safety-critical systems, knowing why a particular memory affected an action may be difficult.

**Catastrophic forgetting is not fully solved.** A task-local memory can help avoid overwriting the same weights, but any learned controller still has limited capacity, and training the controller on new tasks may itself cause interference.

**Offline coverage remains decisive.** If important states never appear in the data, the model cannot reliably discover the corresponding good actions through memory alone.

**Real-world transfer is unproven by this work.** Atari and Meta-World provide useful research benchmarks but do not establish robust deployment in unstructured physical environments. The authors themselves caution that desirable generalization and safety properties are not guaranteed [1].

These are not reasons to dismiss the method. They are the research questions that follow once the basic architecture becomes plausible.

## 29. What this research suggests for today's agents

The broader principle of DT-Mem applies well beyond Atari, but we should be careful about claiming a direct transfer to modern language-model agents.

Consider an agent operating across a repository, a browser, and an internal knowledge base. It needs global skills, temporary working context, and recoverable project-specific facts. Putting every piece of information into model weights is not practical. Repeatedly stuffing an ever-growing history into the prompt is expensive and unreliable.

The attractive alternative is a hierarchy:

```text
Stable learned skills         -> Model parameters
Recent interaction history    -> Context window
Task-specific active state    -> Working memory
Persistent verified knowledge -> External database or retrieval
```

This is a design analogy, not a claim that the ICML 2024 DT-Mem weights can be dropped into any current language-model agent.

It raises useful architectural questions. Should a working memory be updated continuously or only on surprising events? Should it contain learned vectors, symbolic state, or both? How does an agent know whether a memory is stale? Should high-risk actions require retrieval from a verified source rather than an opaque latent state?

A good hybrid system may ultimately use all these memory forms. Neural working memory provides compact, task-conditioned features. External storage provides inspectability and persistence. Learned weights provide transferable competence. Neither one entirely replaces the others.

## 30. A roadmap for a faithful reproduction

Our CPU workshop deliberately stops before expensive policy training. If you want to reproduce the published DT-Mem findings instead of the conceptual mechanism, use the authors' [official repository](https://github.com/luciferkonn/DT_Mem) [2].

Start by pinning the repository commit and documenting the Python, PyTorch, CUDA, dataset, and environment versions. Reconstruct the offline Atari dataset and Meta-World policy trajectories according to the paper. Match the reported architecture, including four Transformer layers, hidden dimension 512, eight attention heads, learned memory projections, slot count, context length, and loss components.

Then verify that the query and key projections produce the expected shapes, the erase/add updates maintain stable memory state, and the action decoder cannot leak future tokens. Check whether memory is reset at episode or task boundaries, and record exactly which parts are persistent across training and inference.

Train a model without memory under matched conditions, then add the memory and change only one factor at a time. Keep task splits fixed. Report validation results at several compute budgets instead of one cherry-picked checkpoint.

For adaptation, freeze the pretrained backbone, add LoRA only to the memory projections specified in the paper, and compare with full fine-tuning and a matched-parameter adapter outside memory. Vary the number of new-task demonstrations while holding optimization steps constant.

Finally, evaluate return through fresh environment rollouts, not solely supervised loss. Keep the held-out task identities, training seeds, normalization, and confidence interval procedure explicit.

That project would be a genuine attempt to reproduce the research. Our companion lab is the smaller first step that makes the memory mechanics understandable before committing to that much compute.

## 31. Conclusion: Remembering is a decision, not just a storage problem

The most useful idea in *Think Before You Act* is deceptively simple.

A model can become better at acting not only by learning more parameters, but by learning **which information to keep available while it acts**.

DT-Mem makes that idea concrete. It augments a trajectory-conditioned Transformer with content-addressed memory, learned updates, and retrieval that participates in action selection. Its ICML 2024 results suggest a promising performance and efficiency tradeoff on selected Atari and Meta-World benchmarks. Its LoRA experiments show that adapting the memory controller can be more economical than changing all of a pretrained model's weights.

Our PyTorch experiments isolate one part of this story. An agent with task-local evidence stored in an addressable memory can answer decisions that are impossible to infer from the context alone. But that advantage is fragile. Reduce capacity and examples collide. Corrupt stored labels and the model remembers the wrong answer. Break the relationship between keys and values and memory contributes almost nothing.

The lesson is not that every Transformer should have a bigger memory matrix.

**The lesson is that intelligent behavior depends on what a system can retrieve, how reliably it can update what it knows, and whether its memories still describe the world in front of it.**

---

## References and further reading

**[1]** Kang, J., Laroche, R., Yuan, X., Trischler, A., Liu, X., and Fu, J. (2024). *Think Before You Act: Decision Transformers with Working Memory*. ICML 2024, Proceedings of Machine Learning Research 235, pages 23001-23021. [Published paper](https://proceedings.mlr.press/v235/kang24b.html) | [arXiv HTML](https://arxiv.org/html/2305.16338) | [PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/kang24b/kang24b.pdf)

**[2]** Kang and collaborators. *DT-Mem official source repository*. [GitHub](https://github.com/luciferkonn/DT_Mem). The tutorial's lab is independent of this repository.

**[3]** Graves, A., Wayne, G., and Danihelka, I. (2014). *Neural Turing Machines*. [arXiv](https://arxiv.org/abs/1410.5401).

**[4]** Sukhbaatar, S., Weston, J., Fergus, R., et al. (2015). *End-To-End Memory Networks*. [arXiv](https://arxiv.org/abs/1503.08895).

**[5]** Chen, L., Lu, K., Rajeswaran, A., et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling*. NeurIPS 2021. [Published paper](https://papers.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html).

**[6]** Janner, M., Li, Q., and Levine, S. (2021). *Offline Reinforcement Learning as One Big Sequence Modeling Problem*. NeurIPS 2021. [Published paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/099fe6b0b444c23836c4a5d07346082b-Abstract.html).

**[7]** Brandfonbrener, D., Bietti, A., Buckman, J., Laroche, R., and Bruna, J. (2022). *When Does Return-Conditioned Supervised Learning Work for Offline Reinforcement Learning?* [arXiv](https://arxiv.org/abs/2206.01079).

**[8]** Hu, E. J., Shen, Y., Wallis, P., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv](https://arxiv.org/abs/2106.09685).

**[9]** Cherepanov, E., Staroverov, A., Yudin, D., Kovalev, A. K., and Panov, A. I. (2023). *Recurrent Action Transformer with Memory*. [arXiv](https://arxiv.org/abs/2306.09459). The DT-Mem paper describes its recurrent-memory baseline using the name RMDT; the linked preprint currently uses this title.

**[10]** Xu, M., Lu, Y., Shen, Y., Zhang, S., Zhao, D., and Gan, C. (2023). *Hyper-Decision Transformer for Efficient Online Policy Adaptation*. [arXiv](https://arxiv.org/abs/2304.08487).

**[11]** Yu, T., Quillen, D., He, Z., et al. (2019). *Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning*. [arXiv](https://arxiv.org/abs/1910.10897).

**Companion code:** The independently implemented PyTorch lab accompanies this article as `dt_mem_working_memory_lab`, including source code, test suite, evaluation harness, saved benchmark JSON, README, and benchmark notes. Results are educational and must not be attributed to the paper's authors.
