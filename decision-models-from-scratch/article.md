# Why Are We Paying Large Language Models to Make Yes-or-No Decisions?

*How JEV, MAYA, and small decision models can change the architecture of AI agents, reduce unnecessary token generation, and turn a fine-tuned Qwen model into a specialized decision engine.*

Imagine that you're designing an AI-powered customer support system.

A customer sends the following message:

> I ordered a laptop last week. It arrived damaged, and I want a replacement.

Before the agent can respond, the system needs to make several decisions.

Is this a refund request or a replacement request? Does the customer need authentication? Should the agent retrieve the order details? Does the company's return policy permit a replacement? Should this conversation be escalated to a human?

A conventional agent might send these questions to a large language model, wait for it to reason, generate structured JSON, parse the response, and then execute the appropriate action.

But look carefully at the problem.

Most of these questions do not require generating language at all. They require selecting an answer from a small, known set of possibilities.

Why use an expensive generative model to produce an answer that could have been represented by a single probability distribution?

That is the idea behind typed decision models, sometimes called System One models.

Rather than asking a language model to generate the next action as text, we train a smaller model to directly estimate the probability of each permissible action. The application can then execute a decision, request additional information, or delegate uncertain cases to a more capable model.

This is not just prompt engineering. It changes how we design the agent itself.

## The research that makes this particularly interesting

There is already evidence that this approach can work inside real agent architectures.

In October 2026, Zhu et al. published [JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs](https://arxiv.org/abs/2610.21412). The researchers trained a Qwen3-4B-based decision model using states encountered by a customer-service agent, including states reached after incorrect decisions.

Their reported results are particularly relevant to our customer-support architecture:

- **Reported agent cost reduction:** 81.1–89%
- **Reported speed improvement:** up to 2.5×

On tasks different from its training distribution, their AgentDecision-4B model achieved 91.5% task success when selecting tools, compared with 82% for Jev, its teacher, under the paper's evaluation conditions.

This gives us a much more interesting research question:

Can we train a small language model to make better decisions than a general-purpose decision model within our own application?

The answer is potentially yes. But the training data, decision architecture, calibration, and agent-level evaluation matter enormously.

Another recent study, [Maya a1](https://arxiv.org/abs/2611.09803), exposes why simply fine-tuning a classifier is not enough. Its researchers found that the underlying Laya checkpoint changed its selected answer on 25.8% of evaluated questions when candidate answers were reordered. Their parallel-layout experiment reduced this instability to 0.5%.

These two papers provide the foundation for our investigation: building a decision model that is not only inexpensive, but also reliable enough to control software.

**[Insert Figure 1: 02-agent-success.png]**

*Figure 1. Reported task success under the JevAgentHarness evaluation conditions.*

## 1. Where decision models fit in a customer-support architecture

The first architectural change is to stop treating the LLM as the controller of absolutely everything.

Instead, separate the system into three responsibilities:

- Decision models: Interpret ambiguous information, classify intent, select tools, score risk and recommend next actions.
- General-purpose LLMs: Handle complex reasoning, unfamiliar situations, explanations and natural-language conversations.
- Deterministic software: Enforce business policies, verify identity, validate permissions and execute approved transactions.

**[Insert Figure 2: 01-support-architecture.png]**

*Figure 2. Proposed hybrid customer-support architecture. Policy and authorization remain deterministic.*

Consider the customer requesting a replacement. The decision model might produce the following distribution:

**Illustrative decision output: Which action should the agent take next?**

- Look up the order: 91%
- Request clarification: 4%
- Escalate to a human: 3%
- Explain the return policy: 2%


The model does not need to generate an explanation of why it chose the order-lookup tool.

The surrounding application already knows what that tool does, what arguments it accepts, and whether the customer is authorized to use it.

This distinction is the key to making the architecture efficient and safe.

## 2. What exactly is a decision model?

Mathematically, a typed decision model estimates:

**pθ(aᵢ | s, q, A)**

Where:

- *s* is the current state of the application.
- *q* is the question being asked.
- *A* is the set of permissible candidate answers.
- *aᵢ* is a particular candidate.
- *θ* represents the learned model parameters.

Given a state, question and candidate set, the model returns:

**p = (p₁, p₂, …, pₖ)**

With:

**Σᵢ pᵢ = 1 (i = 1 to K)**

This differs from a conventional fixed-label classifier in an important way. The question and candidate answers can be supplied dynamically at inference time.

For example, the same model can be asked to classify a support ticket, determine whether a response is grounded in company policy, or select the next available tool.

However, generalizing to new questions and new candidate sets is a learned capability, not something guaranteed by the API design.

### The three decision primitives

Jev and compatible open implementations expose three particularly useful types of questions.

**Choice: select a category**

Example: Which department should handle this ticket?

Billing

Technical support

Returns

Account

Output: a probability distribution over the available categories.

**Noul: estimate the probability of a boolean statement**

Example: Does this customer need to be transferred to a human agent?

Output: *P(true)*, with the probability of false equal to its complement.

**Score: evaluate an ordered rubric**

Example: How urgent is this support request?

Output: probabilities over ordered severity levels, optionally summarized by their expected score.

These three primitives cover a remarkable number of the decisions found inside modern agent workflows.

## 3. Implementation scope

The investigation has two tracks: previously reported CPU experiments and a proposed Qwen3 fine-tuning recipe.

The main implementation begins with `Qwen/Qwen3-0.6B`, rather than requiring a massive language model. This makes it possible to investigate decision models on relatively modest hardware.

The implementation below is an educational Qwen fine-tuning recipe; the offline experiments and the proposed Qwen training are distinct.

The remainder of the article explains the research and training architecture in detail.

## 4. How JEV was actually built

On September 15, 2026, Diogo Almeida, founder of TypeSafe AI, introduced Jev and the company's System One model architecture.

The company's central design decision was to optimize a model specifically for software automation instead of natural-language generation.

According to its official announcement, Jev incorporates three major developments:

- An architecture designed for structured decisions.
- Parallel sampling rather than conventional sequential text generation.
- A training methodology called Reinforcement Learning for Calibrated Decisions (RLCD).

TypeSafe describes the model as returning probabilities and typed values that application software can consume directly.

However, there is an important distinction between what TypeSafe has publicly disclosed and what we can independently reproduce.

The exact proprietary model architecture, complete training dataset, parameter count, and detailed RLCD implementation have not been publicly established in the sources examined here.

Therefore, we should not pretend that we know precisely how to reproduce Jev's internal architecture.

What we can do is reproduce its observable computational interface, using techniques supported by open research.

### The architectural difference

**Conventional generative LLM**

Autoregressive decoding generates successive tokens. Structured output constraints can make the format reliable, but decoding still occurs.

**Typed decision model**

Candidate probabilities are read directly from model scores. The application constructs the structured response.

**[Insert Figure 3: 03-decision-readout.png]**

*Figure 3. Generative decoding versus direct candidate scoring.*

This does not mean decision models perform no computation on input tokens.

They still have to process the prompt, context, candidate descriptions and relevant application state. What they avoid is unnecessary autoregressive answer generation.

That distinction is fundamental to understanding their actual performance advantage.

## 5. Why decision models consume fewer tokens

Suppose we build a customer-service agent that makes four intermediate decisions before responding to a customer.

Each intermediate LLM request contains approximately 800 input tokens and generates 80 output tokens. The final response requires another 800 input tokens and 160 generated tokens.

Our conventional architecture consumes:

**Input tokens = 4 × 800 + 800 = 4,000**

**Output tokens = 4 × 80 + 160 = 480**

That is 4,480 frontier-model tokens per customer interaction.

Now introduce a local decision model that handles 72% of intermediate decisions, sending the remaining 28% to the larger LLM.

The expensive model's expected token consumption becomes:

**Input tokens = 4 × 0.28 × 800 + 800**

**Input tokens = 1,696**

And:

**Output tokens = 4 × 0.28 × 80 + 160 = 249.6**

This represents approximately 56.57% fewer frontier-model tokens under these assumptions.

**Illustrative token budget (100,000 interactions/month).**

- LLM-only: 4,480 tokens per interaction.
- Hybrid: 1,945.6 expected tokens per interaction.
- Frontier-model tokens avoided: 56.57%, or 253.44 million per month.

Assumes a fixed final response budget of 800 input and 160 output tokens. Excludes local-model computation, cache discounts, retries, networking and serving costs.

**[Insert Figure 4: 04-token-budget.png]**

*Figure 4. Illustrative frontier-model token budget, excluding local computation.*

There is a catch, however.

If the small model processes the same large context and then frequently escalates to a larger model, total computation can increase even while expensive API token consumption decreases.

Therefore, a good decision-model architecture should use compact, relevant application state rather than repeatedly forwarding the entire conversation and all tool outputs.

This connects directly to earlier research such as [FrugalGPT](https://arxiv.org/abs/2305.05176) and [RouteLLM](https://arxiv.org/abs/2406.18665), which investigate how to allocate requests among models with different cost and capability profiles.

## 6. The history of decision models did not begin with JEV

Although the System One terminology became popular with Jev, the underlying scientific ideas have existed for decades.

**1950: Brier scores**

A mathematical framework for evaluating the quality of probabilistic predictions.

**1970: Classification with rejection**

Chow investigates when a classifier should abstain instead of making an unreliable prediction.

**1991: Mixture of experts**

Jacobs and colleagues investigate systems where different models specialize in different regions of a problem.

**2001–2016: Conditional computation and distillation**

Researchers develop mechanisms for transferring knowledge into smaller models and executing only necessary computational components.

**2017: Neural network calibration**

Guo et al. demonstrate that modern neural networks can be accurate while producing unreliable confidence estimates.

**2023: FrugalGPT**

Researchers investigate cascades of language models to reduce inference expenditure.

**2024: RouteLLM and agent benchmarks**

Learned routers allocate requests across language models, while τ-bench evaluates stateful tool-using agents.

**2026: JEV, Laya, MAYA and open decision models**

Typed probabilistic interfaces, direct logit readouts, specialized training and confidence-based delegation converge into a practical agent architecture.

Jev's significance is not that classification suddenly became possible.

Its significance is that classification, probabilistic decision theory, pretrained language understanding, and application-oriented inference have been combined into a general-purpose decision interface.

## 7. How Laya and MAYA were built

To understand how to build our own decision model, Laya is particularly interesting because its implementation is publicly accessible.

### Laya architecture

The classic English Laya checkpoint uses:

- ModernBERT-large as a bidirectional language encoder.
- Approximately 395 million backbone parameters.
- A separately trained decision head.
- Two additional transformer layers in that head.
- Option-marker scoring.
- An act-or-escalate prediction mechanism.

The complete English checkpoint contains approximately 421 million parameters.

**[Insert Figure 5: 05-laya-architecture.png]**

*Figure 5. Simplified Laya architecture.*

Rather than predicting a fixed class through an ordinary classification head, Laya scores representations associated with the supplied answer candidates.

This enables the same model to answer different questions with different candidate sets.

### Then researchers discovered something interesting

On November 6, 2026, Mohammed Nayeem Khan and Logesh L published [Maya a1: Do the Options Read the Model, or Does Their Position?](https://arxiv.org/abs/2611.09803)

They investigated whether the order of candidate answers influences the decision produced by Laya.

Suppose we provide a model with:

**Original ordering:** A. Refund; B. Replacement; C. Technical support; D. Account recovery. **Selected: Refund.**

**Reordered options:** A. Account recovery; B. Replacement; C. Technical support; D. Refund. **Selected: Replacement.**

The evidence and question remain identical. Only the answer positions change.

A reliable unordered-choice model should not behave differently merely because the correct answer moves from position A to position D.

The researchers tested 951 questions and reported a 25.8% question-level answer-flip rate for the original Laya checkpoint.

They then fine-tuned it using 2,818 teacher-verified training examples.

The most important result is that the problem was not solved through additional model parameters.

It was addressed by improving how the model encountered and represented candidate answers.

**[Insert Figure 6: 06-maya-order.png]**

*Figure 6. Reported answer-flip rates in the MAYA a1 study.*

### The architectural solution: permutation equivariance

Suppose our candidate scoring function is:

**f(s, q, A) = [z₁, z₂, …, zₖ]**

Let *P* represent a permutation of the candidate answers.

We want:

**f(s, q, PA) = P f(s, q, A)**

In other words, rearranging the options should rearrange their scores, without changing the score associated with each underlying answer.

This property is called permutation equivariance.

A straightforward implementation is to score each candidate through the same learned function:

**zᵢ = gθ(s, q, aᵢ)**

Then normalize across the candidates:

**pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)**

If the scoring function has no access to the candidate's arbitrary position, the scores follow the candidates when they are reordered.

There are still complexities involving identical options, contextual interactions, tokenization, numerical precision, and deterministic tie-breaking. But the mathematical principle provides a much stronger foundation than merely hoping that a sequential model learns to ignore position.

MAYA's parallel-layout experiment applies a related principle by sharing position identifiers across option segments and restricting cross-option attention.

## 8. I built a small experiment to investigate MAYA's findings

I wanted to test whether this positional shortcut could emerge even in a very small neural network.

The offline companion experiment generates 6,000 synthetic decision states with four candidate answers per state.

We deliberately introduce a biased training distribution in which the correct answer frequently appears first.

We train three models:

- A scoring network allowed to learn position-specific biases.
- The same network trained with randomized candidate order.
- A shared scoring network without position-specific parameters.

Then we evaluate each held-out question under all 24 possible permutations of its four options.

Actual companion-lab measurements. These are synthetic experiments, not evaluations of the published MAYA or Laya checkpoints.

**[Insert Figure 7: 07-synthetic-order.png]**

*Figure 7. Synthetic experiment: answer-flip rates across all candidate permutations.*

The most interesting part is that the original biased scorer achieved **95.7% accuracy on the canonical ordering**.

Once we evaluated it across different candidate arrangements, its mean accuracy fell to 87.38%.

The position-free model achieved 89.7% accuracy across every tested permutation.

This demonstrates why an apparently better validation result can hide undesirable behavior.

The model that looks best under the familiar evaluation format may be exploiting a shortcut that will not survive changes in the application's actual inputs.

## 9. Building a decision model from Qwen without training anything

Here is where the implementation becomes especially interesting.

We do not necessarily need to invent a completely new architecture.

An existing language model already calculates a probability distribution over its vocabulary whenever it predicts its next token.

Suppose we ask a Qwen model:

```text
Customer message:
I was charged twice for my laptop.
Question:
Which department should handle this request?
A. Technical support
B. Billing
C. Returns
D. Account recovery
Answer:
```

Instead of calling:

```python
model.generate(...)
```

We can directly inspect the final-position logits.

```python
outputs = model(
    **inputs,
    use_cache=False,
    logits_to_keep=1
)
logits = outputs.logits[:, -1, :]
```

The tensor contains one logit for each vocabulary token.

We only need the logits associated with our four answer letters.

```python
option_ids = [
    tokenizer.encode(
        letter,
        add_special_tokens=False
    )[0]
    for letter in ["A", "B", "C", "D"]
]
option_logits = logits[:, option_ids]
probabilities = torch.softmax(
    option_logits,
    dim=-1
```

)

In a real implementation, we must verify that every answer marker is represented by exactly one token and that the chosen token representation matches the prompt format.

But the essential idea is surprisingly simple.

We have converted a generative language model into a typed decision engine without requiring an autoregressive answer-generation loop.

### Why this works

The language model's original output layer computes:

**z = Wh + b**

Where:

- *h* is the final hidden representation.
- *W* is the language-model output matrix.
- *z* contains vocabulary logits.

We can extract the rows corresponding to candidate answer tokens:

**W_A**

And obtain:

**z_A = W_A h + b_A**

Finally:

**p_A = softmax(z_A)**

The model is no longer asked to explain its answer. We directly use the knowledge encoded in its hidden representation.

### Research supporting this approach

In [Releasing the Inner Jev](https://arxiv.org/abs/2610.17205), Geuter, Cheng and Alvarez-Melis (2026) evaluated this letter-readout technique with open language models.

Their experiments showed that existing instruction-tuned models already possess substantial typed decision-making capabilities. LoRA adaptation improved performance further, and their fine-tuned Gemma 4 31B exceeded Jev on selected evaluation suites.

This is compelling evidence that decision models do not necessarily need to be pretrained from scratch.

However, it is not evidence that an untrained 600-million-parameter model will match a proprietary frontier model on arbitrary questions.

That depends on the task, training distribution, base model and evaluation protocol.

## 10. Training your own Qwen decision model

For a first custom implementation, I would use the following architecture.

**[Insert Figure 8: 08-qwen-training.png]**

*Figure 8. Proposed LoRA training pipeline for Qwen.*

### Step 1: Define your decision space

Do not begin by collecting random instructions and responses.

Begin by documenting what decisions your application must make.

For customer support, a decision taxonomy might look like this:

- Classify customer intent: choice.
- Select the next tool: choice.
- Determine whether human review is necessary: boolean.
- Determine whether a response follows company policy: boolean.
- Estimate issue severity: ordinal score.
- Determine whether more information is required: boolean.
- Select the relevant knowledge-base article: choice.
- Determine whether an agent has completed its task: boolean.

Some of these decisions should not require machine learning at all.

For instance, whether a customer has successfully authenticated should come from the authentication service, not from a model interpreting the customer's statement.

Use models for ambiguous semantic decisions and code for facts that can be evaluated deterministically.

### Step 2: Construct your training dataset

Each training example should contain a complete decision context.

```json
{
  "state": {
    "message": "My headphones arrived damaged.",
    "authenticated": true,
    "order_found": true,
    "replacement_eligible": true
  },
  "question": "What should the agent do next?",
  "options": [
    {
      "id": "refund",
      "text": "Propose an eligible refund"
    },
    {
      "id": "replacement",
      "text": "Propose an eligible replacement"
    },
    {
      "id": "clarification",
      "text": "Ask whether the customer wants a refund or replacement"
    },
    {
      "id": "escalate",
      "text": "Refer to a human specialist"
    }
  ],
  "answer": "clarification"
}
```

Notice that the damaged product alone does not tell us whether the customer prefers a refund or replacement.

Training on overly simplistic examples would teach the model to make assumptions that might be incorrect.

Good training data should include:

- Ordinary successful cases.
- Ambiguous requests.
- Missing information.
- Conflicting information.
- Incorrect previous tool calls.
- Tool failures.
- Out-of-distribution requests.
- Cases where the correct behavior is to abstain.
- Cases where no available option is appropriate.

An explicit `none_of_the_above` or `ask_for_clarification` action can be particularly valuable.

### How much data do you need?

The following are reasonable starting budgets for experiments, not guaranteed sample-complexity requirements.

- One narrowly defined classification task: 500–2,000 examples.
- Several support decisions: 2,000–10,000 examples.
- Multi-stage custom agent: 10,000–50,000 decision states.
- Broad, general-purpose decision model: potentially hundreds of thousands.

The required quantity depends on task diversity, label quality, how much relevant knowledge exists in the base model, and the reliability target.

I would start small, build learning curves, and collect additional examples based on the actual failure distribution.

## 11. Knowledge distillation: learning from a larger decision model

An effective way to construct a small custom decision model is to use a more capable model as a teacher.

This connects to the knowledge-distillation research of Hinton, Vinyals and Dean (2015).

Instead of recording only the teacher's selected answer, we can record its complete candidate distribution.

For example:

```json
{
  "replacement": 0.78,
  "refund": 0.12,
  "clarification": 0.08,
  "escalate": 0.02
}
```

These probabilities communicate more information than a simple hard label.

The student learns not only that replacement is the preferred answer, but also how the teacher differentiates plausible and implausible alternatives.

The distillation objective is:

**L_distill = KL(p_T ‖ p_S)**

Where *p_T* is the teacher distribution and *p_S* is the student distribution.

We can combine it with supervised learning from verified labels:

**L = α L_hard + (1 − α) L_distill**

For practical applications, the teacher must not become our unquestionable source of truth.

If the teacher disagrees with a verified business policy or human annotation, the verified label should take precedence.

The JevAgentHarness researchers used a related strategy, incorporating teacher distributions when consistent with correct answers and relying on gold labels when the teacher selected an incorrect option.

That is one reason a specialized student can potentially outperform its teacher within a restricted application.

## 12. Why calibration is just as important as accuracy

Imagine that our small decision model correctly handles 90% of requests.

Is it ready for production?

Not necessarily.

If the model knows which 10% of cases it is likely to get wrong, we can escalate those cases to a larger model or human operator.

If it reports 99% confidence on incorrect answers, the architecture becomes much more dangerous.

A well-calibrated classifier should satisfy approximately:

**P(ŷ = y | p̂ = c) = c**

For predictions assigned confidence 0.8, approximately 80% should be correct over an appropriate population.

### Cross-entropy

For a target distribution *y* and prediction *p*:

**L_CE = −Σᵢ yᵢ log(pᵢ)**

Cross-entropy penalizes the model for assigning low probability to the correct outcome.

### Brier score

**L_Brier = Σᵢ (pᵢ − yᵢ)²**

The Brier score measures the squared difference between predicted probabilities and observed outcomes.

A practical training objective can combine both:

**L = L_CE + λ L_Brier**

Both encourage probabilistically meaningful predictions, although minimizing these objectives during training does not guarantee perfect calibration on new data.

### Temperature scaling

Guo et al. (2017) demonstrated that a simple post-training temperature parameter can improve neural-network calibration.

**pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)**

**Temperature scaling example:** at T = 1, illustrative logits (3.0, 1.7, 0.5, −0.8) give approximately 72.62%, 19.79%, 5.96%, and 1.62% for A through D. Temperature changes confidence without changing the highest-scoring answer.

**[Insert Figure 9: 09-temperature.png]**

*Figure 9. Illustrative temperature scaling for the supplied logits.*

In the offline experiment, temperature scaling actually worsened held-out expected calibration error from approximately 1.27% to 1.85%.

I have deliberately retained this result because it illustrates an important point: fitting a temperature on calibration data does not guarantee that every calibration metric improves on every independent test set.

Always evaluate it.

## 13. What is RLCD, and do we actually need reinforcement learning?

Reinforcement Learning for Calibrated Decisions is the training terminology used by TypeSafe for Jev.

The proprietary implementation is not fully disclosed.

Laya, however, provides an open implementation involving reinforcement-style optimization and proper scoring rules. Its public training tooling also supports ordinary soft cross-entropy as an alternative.

A strictly proper scoring rule encourages a probabilistic predictor to report its true estimated probability distribution rather than an artificially exaggerated confidence.

For a correct outcome *y*, a logarithmic scoring reward can be written:

**R_log(p, y) = log(p_y)**

A negative Brier reward can be written:

**R_Brier(p, y) = −Σᵢ (pᵢ − yᵢ)²**

One possible educational objective is:

**R = λ₁ R_log + λ₂ R_Brier**

This is an illustrative design, not the undisclosed original Jev objective.

For most custom business applications, I would not begin with reinforcement learning.

Start with supervised fine-tuning, reliable labels, option shuffling and calibration.

These are easier to implement, easier to debug, and provide an essential baseline against which reinforcement learning can be evaluated.

Add reinforcement learning when the optimization objective genuinely depends on the consequences of decisions across a multi-step environment.

## 14. Why training inside an agent matters

Suppose an agent has to make ten consecutive decisions.

Each decision is independently correct with probability 0.95.

Under a simplified assumption that all ten must be correct and errors are independent:

**P(task success) = 0.95¹⁰**

**P(task success) ≈ 59.9%**

Even a highly accurate decision model can produce a surprisingly unreliable agent when mistakes accumulate.

**Illustrative example:** 95% per-decision accuracy across 10 required decisions gives 59.87% complete task success. Actual agent failures may be correlated, and agents may recover from earlier mistakes.

**[Insert Figure 10: 10-error-compounding.png]**

*Figure 10. Independent per-decision success model, not a measured agent benchmark.*

This is one of the central findings of JevAgentHarness.

A model that performs well on isolated multiple-choice questions is not necessarily the best model to control an actual agent.

Their approach involved collecting states from running agents, including states reached after mistakes, and training the small model on those states.

The reported training collection included 15,694 decision states across 587 tasks, with 30.9% of the states following errors. The researchers then applied supervised fine-tuning and reinforcement learning inside the agent loop.

For our own customer-support system, this suggests a training workflow:

**[Insert Figure 11: 11-training-loop.png]**

*Figure 11. Proposed trajectory-based training workflow.*

This is considerably more valuable than training exclusively on clean, successful demonstrations.

## 15. Can a 0.6B Qwen model compete with JEV?

For a narrowly defined task, potentially.

For arbitrary general-purpose decisions, we should not assume so.

A particularly interesting November 2026 preprint, [Small Heads, Strong Decisions](https://arxiv.org/abs/2611.15592), studied different decision heads under controlled experimental conditions.

On Qwen3-0.6B, five different readouts achieved broadly similar in-domain accuracy, ranging from 81.3% to 82.0%.

The study also reported that a relatively small cosine-similarity head matched a much larger schema head on its in-domain evaluation while requiring less inference time. Moving to a larger backbone improved generalization to held-out sources.

This suggests a useful engineering principle:

Do not spend weeks designing a sophisticated decision head before testing whether your existing language model already contains the necessary knowledge.

I would investigate the following progression.

**Qwen3-0.6B**

Start with narrow, high-volume routing, classification and simple verification tasks. Test zero-shot letter readout before fine-tuning.

[Model checkpoint](https://huggingface.co/Qwen/Qwen3-0.6B)

**Qwen3-1.7B**

An intermediate experiment for tasks where the smallest model struggles with language understanding or unfamiliar phrasing.

[Model checkpoint](https://huggingface.co/Qwen/Qwen3-1.7B)

**Qwen3-4B**

A stronger candidate for contextual tool selection, policy interpretation and complex agent states. AgentDecision-4B provides a relevant published example.

[Model checkpoint](https://huggingface.co/Qwen/Qwen3-4B)

**Laya / ModernBERT**

An encoder-based alternative for applications dominated by text classification and typed decisions rather than generative capabilities.

[Open implementation](https://github.com/NandhaKishorM/laya)

## 16. The actual Qwen fine-tuning mechanism

The optional companion implementation uses LoRA.

For an original pretrained weight matrix *W*, LoRA introduces a low-rank update:

**W′ = W + (α / r) BA**

The pretrained weights remain frozen while the small matrices *A* and *B* are optimized.

In Python:

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name
)
model = AutoModelForCausalLM.from_pretrained(
    model_name
)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
```

Instead of training the model to produce a lengthy assistant response, we optimize the answer-candidate distribution.

```python
import torch
import torch.nn.functional as F
def decision_loss(logits, targets):
    log_probs = F.log_softmax(
        logits,
        dim=-1
    )
    probs = log_probs.exp()
    cross_entropy = -(
        targets * log_probs
    ).sum(dim=-1).mean()
    brier = (
        probs - targets
    ).square().sum(dim=-1).mean()
    return cross_entropy + 0.1 * brier
```

This implementation can learn from either hard labels or teacher-provided soft distributions.

During training, we also randomize the candidate order and move the target label accordingly.

```python
import random
def shuffle_example(example):
    options = example["options"].copy()
    random.shuffle(options)
    target = next(
        i for i, option in enumerate(options)
        if option["id"] == example["answer"]
    )
    return options, target
```

Always preserve stable candidate IDs.

If we shuffle the candidate texts but do not update their target indices, we corrupt the training data.

### One subtle problem: discarded vocabulary probability

Suppose the model assigns these probabilities to four answer letters:

```text
A = 0.004
B = 0.001
C = 0.001
D = 0.004
```

Only 1% of the model's full vocabulary probability mass belongs to the permitted answers.

Renormalizing those four values produces:

```text
A = 40%
B = 10%
C = 10%
D = 40%
```

The conditional distribution looks meaningful, but the model was barely assigning probability to the answer format at all.

Therefore, I recommend measuring:

**m_A = Σᵢ∈A p_vocab(i)**

Alongside the restricted candidate probabilities.

Low option mass can be an additional reason to defer a decision. It is not, by itself, a universally reliable uncertainty measure, but it provides useful diagnostic information.

Include this calculation in the Qwen implementation when evaluating candidate confidence.

## 17. Designing the confidence gate

The most useful production model is not necessarily the one with the highest classification accuracy.

It is the one that can make enough reliable decisions without consuming unnecessary resources.

We define coverage as:

**C(t) = P(c(x) ≥ t)**

And selective risk as:

**R(t) = P(ŷ ≠ y | c(x) ≥ t)**

Where *t* is our acceptance threshold.

In the offline experiment, a threshold of 0.87 selected using calibration data produced the following held-out results:

- Decisions accepted automatically: 72%.
- Accuracy on accepted decisions: 97.92%.
- Decisions sent to fallback: 28%.
- Acceptance threshold: 0.87.

These are actual results from the small synthetic scoring experiment, not Qwen or Jev production measurements.

For customer support, different decisions should have different thresholds.

For example, an incorrect department classification might be relatively inexpensive to correct, whereas an unauthorized financial transaction may cause serious harm.

An illustrative policy might be:

- Routine ticket classification: accept above validated threshold.
- Knowledge-base selection: accept with relevance checks.
- Customer escalation: prefer recall for serious cases.
- Refund eligibility: require deterministic policy verification.
- Financial transaction: require authorization and approval.

Do not simply set every threshold to 0.95 and assume that the resulting system is safe.

## 18. How this would work in an AI engineering interview

If an interviewer asks you to design an AI customer-support system, I would describe the architecture in the following manner.

I would separate the decision-making layer from the generative layer.

Instead of calling a large language model at every step, I would use a small specialized decision model to classify customer intent, select permissible tools, identify missing information, and determine when escalation is required.

For decisions with a bounded answer space, the model can score candidate actions directly using one forward pass, rather than generating and parsing an autoregressive response.

I would establish a fixed-label classifier and a zero-shot Qwen decision model as baselines, then fine-tune the smaller model using LoRA on historical support cases and agent trajectories, including failure and recovery states.

I would calibrate the model on independent data and use decision-specific thresholds to determine when a larger language model or a human should take over.

Importantly, I would not allow the decision model to authorize refunds or other sensitive transactions. A deterministic policy and permission layer would validate the proposed actions before execution.

Finally, I would evaluate the complete system using task completion rate, policy violations, escalation rate, selective accuracy, p95 latency, and cost per successfully resolved customer request, rather than relying only on individual classification accuracy.

That answer demonstrates knowledge of model training, inference optimization, system architecture, evaluation methodology, operational safety and production engineering.

## 19. Open-source projects worth investigating

[Laya](https://github.com/NandhaKishorM/laya)

Open non-autoregressive decision models with specialized scoring heads, calibration, multilingual support and fine-tuning tools. Particularly useful for understanding encoder-based decision architectures.

[MAYA a1](https://arxiv.org/abs/2611.09803)

Research investigating option-order sensitivity and how training augmentation and architectural changes can address it.

[JevAgentHarness](https://github.com/JevAgentHarness/JevAgentHarness)

Highly relevant to customer-support architecture. Investigates delegation, confidence gating, agent trajectories and specialized decision-model training.

[AgentDecision-4B](https://huggingface.co/JevAgentHarness/AgentDecision-4B)

A Qwen3-4B LoRA model trained specifically for decisions inside agent environments. Check its noncommercial license before considering production use.

[Qev](https://github.com/QiqianFu/Qev)

Provides training infrastructure and checkpoints for converting Qwen models into typed decision engines.

[AnyJev](https://github.com/nokia-applied-research/AnyJev)

Explores extracting typed decisions from existing language models without necessarily retraining them.

[JevAny](https://simplejev.org/JevAny/docs/TRAINING.html)

Provides recipes for training models using direct answer-token readouts and learned pointer heads.

[RouteLLM](https://github.com/lm-sys/RouteLLM)

Useful for understanding a related problem: selecting which language model should handle a request according to its anticipated capability and cost.

These projects provide several different architectural starting points. They are not interchangeable, and their licenses, evaluation datasets, and production readiness should be assessed individually.

## 20. An important recent development: OpenAI's Decisions API

This research direction is no longer limited to experimental open-source projects.

OpenAI also provides a generally available Decisions API for evaluating text, images and structured inputs against bounded questions.

It supports predicates, choices and ordered scores, and its documentation describes processing independent questions together against shared evidence.

This reinforces the broader architectural shift toward treating AI as a probabilistic component inside ordinary software, rather than requiring every AI interaction to take the form of a conversation.

However, hosted decision APIs and locally trained decision models solve somewhat different operational problems.

A hosted model can provide broad capability without maintaining training infrastructure.

A specialized local model offers more control over data, domain adaptation, serving infrastructure and model behavior.

For a startup building a high-volume AI product, I would evaluate both approaches before committing to either.

## Conclusion

The most interesting implication of decision models is not simply that we can make language models smaller.

It is that many applications have been using the wrong computational interface.

A customer-support agent does not always need a generated explanation. Sometimes it needs a probability, an action, a category or a decision to request assistance.

Once we recognize that distinction, we can design a different kind of AI architecture: small models handling routine decisions, larger models handling difficult reasoning, and deterministic software governing execution.

That architecture can be cheaper and more responsive, but its reliability still depends on the quality of its training data, calibration, robustness, and end-to-end evaluation.

Research and implementation limitations: I verified the public descriptions of JEV, Laya, MAYA and the relevant recent research. Jev's proprietary internals remain undisclosed. The synthetic scoring, calibration, customer-support baseline, policy checks and token-budget simulation were executed. The optional Qwen training recipe was syntax-checked but not trained because its model weights and required libraries were unavailable in the execution environment. I have not established that our custom model matches JEV in performance, and the literature review is selective rather than exhaustive.

The next meaningful research experiment would be to train Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B on identical customer-support trajectories and compare them against Jev, Laya and a conventional generative agent using the same task-success, risk, latency and cost metrics.

## References and further reading

These are the most relevant sources for understanding and building decision models, organized by research direction.

### Foundational research

1. [Brier, G. W. (1950). Verification of Forecasts Expressed in Terms of Probability.](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2)

2. [Chow, C. K. (1970). On Optimum Recognition Error and Reject Tradeoff.](https://doi.org/10.1109/TIT.1970.1054406)

3. [Jacobs, R. A., et al. (1991). Adaptive Mixtures of Local Experts.](https://doi.org/10.1162/neco.1991.3.1.79)

4. [Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural Network.](https://arxiv.org/abs/1503.02531)

5. [Guo, C., et al. (2017). On Calibration of Modern Neural Networks.](https://arxiv.org/abs/1706.04599)

6. [Geifman, Y., and El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks.](https://arxiv.org/abs/1705.08500)

7. [Mozannar, H., and Sontag, D. (2020). Consistent Estimators for Learning to Defer to an Expert.](https://proceedings.mlr.press/v119/mozannar20b.html)

8. [Ross, S., Gordon, G., and Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.](https://arxiv.org/abs/1011.0686)

### Efficient language-model inference and adaptation

9. [Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.](https://arxiv.org/abs/2106.09685)

10. [Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.](https://arxiv.org/abs/2305.14314)

11. [Chen, L., Zaharia, M., and Zou, J. (2023). FrugalGPT.](https://arxiv.org/abs/2305.05176)

12. [Ong, I., et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data.](https://arxiv.org/abs/2406.18665)

13. [Warner, B., et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference.](https://arxiv.org/abs/2412.13663)

14. [Qwen Team (2025). Qwen3 Technical Report.](https://arxiv.org/abs/2505.09388)

### Robust decision-making and agent evaluation

15. [Zheng, C., et al. (2023). Large Language Models Are Not Robust Multiple Choice Selectors.](https://arxiv.org/abs/2309.03882)

16. [Pezeshkpour, P., and Hruschka, E. (2023). Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions.](https://arxiv.org/abs/2308.11483)

17. [Yao, S., et al. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.](https://arxiv.org/abs/2406.12045)

### Recent typed-decision research

18. [Almeida, D. (2026). Introducing System One Models and Jev. Official company announcement.](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

19. [ConvAI Innovations (2026). Laya model card and architecture.](https://huggingface.co/convaiinnovations/laya)

20. [Geuter, J., Cheng, A., and Alvarez-Melis, D. (2026). Releasing the Inner Jev: A Simple Recipe for Open Decision Models.](https://arxiv.org/abs/2610.17205)

21. [Zhu, K., et al. (2026). JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs.](https://arxiv.org/abs/2610.21412)

22. [Khan, M. N., and Logesh L. (2026). Maya a1: Do the Options Read the Model, or Does Their Position?.](https://arxiv.org/abs/2611.09803)

23. [Rastogi, R. (2026). Small Heads, Strong Decisions: A Controlled Study of Typed-Decision Models.](https://arxiv.org/abs/2611.15592)

24. [TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models.](https://arxiv.org/abs/2610.11392)

25. [Our Own Jev: Typed Decisions in One Forward Pass, from an In-House Judge to Open LoRA Models.](https://arxiv.org/abs/2610.29052)

26. [OpenAI (2026). Build More Responsive Apps with the Decisions API.](https://openai.com/index/decisions-api/)
