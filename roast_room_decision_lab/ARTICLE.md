# Seven AI Advisors Debated One Business Decision. The Real Question Is Whether Debate Makes AI More Accurate.

### Inside Roast Room's Claude Code architecture, specialized agents, weighted decision rules, and a reproducible experiment on when multi-agent deliberation helps and when it turns into expensive groupthink

A founder has three options.

Raise prices for everyone. Raise them only for new customers. Keep prices where they are.

Each option has an attractive story. Raising everybody's prices increases revenue immediately, but might increase churn. Raising only for new customers protects existing relationships, but leaves money on the table. Doing nothing preserves stability, but could mean continuing to undercharge.

You could ask an AI assistant to recommend one. You would probably get a balanced answer, an inventory of pros and cons, and a confident closing paragraph.

Now imagine something different. Seven independently instructed AI advisors evaluate the same choice. One focuses on customer demand. Another focuses on unit economics. Another worries about cash running out before the strategy starts working. They disagree, receive each other's arguments, revise their scores, and produce a final resolution with a dissenting minority report.

That is the idea behind [Roast Room](https://github.com/proxient/learn-roast-room), the Claude Code project we are studying.

The more interesting question is not whether seven personalities can produce an entertaining debate. They can. It is whether turning one model into a council of sub-agents produces better decisions than a simpler, cheaper approach.

That question is surprisingly difficult. All seven advisors can inherit the same mistaken premise. They can become convinced by a persuasive but incorrect argument. A weighted average can conceal a catastrophic risk. A confident consensus can emerge because agents have started copying each other's judgments rather than verifying evidence.

This tutorial explores those problems from the inside out. We will examine the actual repository, implement its most important scoring ideas in Python, construct synthetic business decisions with known outcomes, and run seven controlled configurations to measure choice quality, regret, disagreement, false approvals, and computational cost.

**What we measured:** Across five seeds and 600 synthetic decisions per seed, a one-seat generalist selected the best option 54.0% of the time. Seven independently noisy specialists reached 81.7%. Introducing a shared framing bias reduced the seven-seat result to 60.2%. Simulated discussion that merely moved scores toward the group average reduced disagreement, but did not improve accuracy. Providing better independent evidence produced the strongest option-selection result at 86.6%.

These are original educational simulations, **not measurements of Claude, Roast Room's production behavior, or any actual company's decisions**. The design is intentionally small enough to audit, not realistic enough to certify business advice.

---

## Contents

1. What Roast Room actually is
2. Why a council differs from seven copies of the same prompt
3. The repository architecture
4. The Chair and the seven specialist seats
5. The first round: independent positions
6. Debate, ratification, and the information problem
7. The scoring system in executable terms
8. Why the risk veto matters
9. The unanimity trap
10. The local room and event-driven UI
11. Provenance, quotation checking, and its limits
12. Token costs and the 21-agent architecture
13. Research on multi-agent debate
14. The independent benchmark we will build
15. Generating decisions with known answers
16. Simulating specialists without LLMs
17. Reproducing relevance-weighted aggregation
18. Implementing the ruin cap
19. Seven benchmark configurations
20. Accuracy is not enough: utility regret
21. Our measured results
22. Why correlated errors defeat multi-agent ensembles
23. How debate can reduce disagreement without increasing truth
24. Better evidence versus more discussion
25. A worked example from the executable lab
26. Running the real Roast Room
27. What I would improve in the original design
28. Threat model and practical deployment checklist
29. Limitations and experimental honesty
30. Final lessons and further experiments
31. References and source links

---

## 1. What Roast Room actually is

The linked repository, [`proxient/learn-roast-room`](https://github.com/proxient/learn-roast-room), is a public GitHub fork of the [original Roast Room by `nateherkai`](https://github.com/nateherkai/roast-room). That distinction matters for attribution: the linked project is a fork, not independent evidence of a separately invented framework.

The repository's own README describes a council of seven simulated advisors based on public writing, talks, interviews, and business frameworks associated with well-known operators. These are **not the actual individuals**, and the repository explicitly states that they do not speak for those people or represent endorsements.

The available seats examine different dimensions of a business decision:

- **Bezos:** long-term strategy, customer orientation, and whether a decision can be reversed.
- **Paul Graham:** whether a product has real demand and growth potential.
- **Alex Hormozi:** offers, pricing, acquisition, and commercial viability.
- **Gary Vaynerchuk:** attention, distribution, and marketing opportunities.
- **Dan Martell:** staffing, delegation, and the founder's time.
- **Gino Wickman:** operations, accountability, and organizational execution.
- **Warren Buffett:** capital allocation, downside risk, and survivability.

The system is primarily a **Claude Code skill plus project-local sub-agents**. A separate Node.js web application visualizes what happens. You can use the council without opening the web interface.

This is an important architectural choice. The web app is not a second AI that reads your decision and performs the analysis. In its default viewer configuration, it watches files written by the running council and displays them. There is also an optional direct Anthropic API runner, but that is a separate, explicitly enabled mode.

So the core system is not a giant application framework. It is a carefully specified agent protocol, organized as files.

**Primary sources:** [Fork README](https://raw.githubusercontent.com/proxient/learn-roast-room/main/README.md), [Roast Room skill](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/SKILL.md).

## 2. Why a council differs from seven copies of the same prompt

A naive multi-agent application calls a model seven times, collects seven answers, and reports the majority opinion. That is an ensemble, but it is not automatically a useful expert system.

To understand the difference, consider what makes an ensemble effective in classical machine learning. If several classifiers make errors on different examples, combining them can cancel some errors. If they make exactly the same errors, adding more classifiers provides little benefit. Seven identical wrong answers are not stronger evidence than one wrong answer.

The same principle applies to language model agents, except that the failure modes are more subtle. Agents can share model weights, training data, prompt assumptions, external sources, and the exact same framing of the question. Their prose may differ dramatically while their underlying factual mistakes remain almost perfectly correlated.

Roast Room attempts to create diversity by assigning each seat a different question, a different source-derived knowledge collection, an explicit weight, and a role-specific prompt. The goal is to create **functional specialization** rather than seven theatrical voices.

The distinction is testable. You should ask whether a pricing specialist notices unit-economics errors that a strategy specialist misses. You should not merely ask whether the two outputs sound different.

The same issue appears in scientific multi-agent workflows: dividing the problem by evidence, hypothesis, or specialized tool can be valuable. Dividing it into personalities alone does not guarantee anything. Specialists need different evidence or different error structures, not just distinctive wording.

Our lab will make this explicit by comparing **independent noise** against **shared bias**. The latter allows every simulated specialist to look individually capable while the group fails together.

## 3. The repository architecture

At the top level, the linked repository contains `.claude/`, `app/`, `docs/`, `sessions/`, a `package.json`, and ordinary project files.

The mechanism lives in a handful of distinct areas:

```text
learn-roast-room/
├── .claude/
│   ├── agents/
│   │   ├── council-bezos.md
│   │   ├── council-graham.md
│   │   ├── council-hormozi.md
│   │   └── ...
│   └── skills/
│       └── roast-room/
│           ├── SKILL.md
│           ├── brains/
│           │   ├── jeff-bezos/
│           │   ├── paul-graham/
│           │   └── ...
│           ├── references/
│           │   ├── seats.md
│           │   ├── round-prompts.md
│           │   ├── scoring.md
│           │   ├── report-template.md
│           │   └── live-room.md
│           └── scripts/
│               └── check_quotes.py
├── app/
│   ├── server.mjs
│   ├── lib/
│   └── public/
├── sessions/
├── package.json
└── README.md
```

The most interesting file is not JavaScript. It is [`SKILL.md`](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/SKILL.md). It defines the full workflow, including how the Chair collects the initial brief, how to launch sub-agents, how to write intermediate artifacts, how to handle premature agreement, and how to generate the final report.

The agent definitions are small Markdown files. For example, the Bezos seat declares its own name, available tools, a turn limit, its source material, its decision framework, and a requirement not to impersonate the real person. It has read-oriented tools rather than general unrestricted execution authority.

Meanwhile, the web app uses the Node.js standard library. The root `package.json` specifies `node app/server.mjs` as its start script and requires Node.js 20 or newer. There is no elaborate package dependency tree required just to display the room.

A practical consequence is that the orchestration protocol remains readable and modifiable. Another consequence is that correctness depends heavily on the instructions actually being followed.

**Primary sources:** [Agent definition](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/agents/council-bezos.md), [package.json](https://raw.githubusercontent.com/proxient/learn-roast-room/main/package.json).

## 4. The Chair and the seven specialist seats

The Chair is not meant to be an eighth advisor. Its job is to organize evidence and disagreements, not add another business philosophy.

The skill tells the Chair to obtain a decision brief with a finite set of options. It must include a status quo alternative or an explicit choice not to proceed. That seemingly mundane rule is a valuable defense against a common planning error: a model is asked *how* to execute a proposed action before anyone checks whether executing it is wise.

The Chair gathers the business context, available cash, team capacity, time horizon, success criteria, and the user's current preference. Missing details must be marked as assumptions. It keeps the brief concise so every agent is analyzing the same case.

Next it assigns each seat a relevance weight between one and three. A pricing decision may give the pricing specialist the highest weight. A risky financing decision may give the capital specialist a greater role. Exactly one or two seats receive weight three in the repository's specified protocol.

This is not a claim that any famous person's worldview is inherently worth three times another person's. It is a formal way to say that expertise should matter in proportion to the problem being solved.

The Chair also records the weights and their reasons. That creates a small but important audit trail. If the conclusion is surprising, a reader can ask whether the problem was framed badly, whether the evidence was weak, or whether the weights favored the wrong dimensions.

**The core design insight:** separate evidence generation from decision aggregation. An agent can be excellent at identifying a pricing risk and still be a poor choice to determine the final strategy alone.

## 5. The first round: independent positions

In the opening round, every seat gets the same brief plus its own instructions and selected source files. The skill asks the Chair to spawn the agents in parallel and have each return a structured card.

A card contains the agent's position, numerical scores for every option, its pillar-specific score, confidence, an assessment of irreversible downside risk, a reversibility judgment, reasons, a major risk, an observation that would change its mind, and a concrete action for the coming week.

Notice the last two fields. **What would change my mind** prevents the answer from becoming an unfalsifiable opinion. **What I'd do this week** turns a debate into an experiment.

The agents are instructed to read their own source-derived brain, cite the relevant framework identifiers, and avoid inventing quotations. A specialist can argue strongly for its own dimension because the Chair is responsible for reconciling conflicting perspectives.

This is a better protocol than asking every agent to be balanced. When every specialist tries to produce the perfect balanced answer, their outputs tend to converge before the debate even begins. The useful diversity has already been lost.

However, independence is only partial. The seats are still language-model instances that receive a common brief. They may also share a model family and many learned assumptions. Parallel execution creates separate processing contexts. It does **not** establish statistical independence of their errors.

That distinction explains why our benchmark needs a shared-framing-bias condition, not just independent random noise.

**Primary sources:** [Round prompts](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/references/round-prompts.md), [Anthropic's explanation of Claude Code sub-agents](https://claude.com/resources/articles/steering-claude-code-skills-hooks-rules-subagents-and-more).

## 6. Debate, ratification, and the information problem

After the first round, the Chair identifies the cruxes. A crux is not simply a topic on which advisors disagree. It is a factual assumption, numerical claim, or value tradeoff that explains why they recommend different options.

For example:

> Will an increase in price reduce customer retention enough to eliminate the additional revenue?

That is useful because it can be answered by looking at churn data or running a controlled price test. The statement *Marketing disagrees with Finance* is less useful. It describes a conversation, not a decision boundary.

The second round asks advisors to respond directly to competing arguments, state what they concede, and explain any changed score. The third round is not another open debate. It is a ratification of a proposed resolution: endorse it, endorse with an amendment, or dissent.

The full default mode involves seven seats across three rounds, so its basic seat structure is **21 sub-agent runs**. The skill file estimates a full run at roughly one million tokens. That estimate should not be interpreted as a universal measured bill. Actual token consumption depends on context length, source retrieval, model choice, caching, and what the agents produce.

An important question appears here: what new information enters the process between rounds?

If Round 2 contains only the arguments produced in Round 1, the agents can clarify reasoning but cannot verify a factual claim they all got wrong. Debate can improve the organization of evidence; it cannot manufacture missing external evidence.

If the council receives actual unit economics, customer interviews, experiment results, or a verified cash forecast, the situation is different. The agents now have a reason to update.

This leads to the most important ablation in our experiment: compare **extra discussion** with **better evidence**.

## 7. The scoring system in executable terms

Roast Room computes each option's score as the relevance-weighted average of the individual seat scores. Every seat rates each option from zero to ten.

Suppose four seats score three options:

```python
cards = [
    {'seat': 'product', 'weight': 3, 'scores': [8, 6, 4]},
    {'seat': 'pricing', 'weight': 3, 'scores': [9, 6, 5]},
    {'seat': 'capital', 'weight': 2, 'scores': [6, 7, 8]},
    {'seat': 'operations', 'weight': 1, 'scores': [8, 7, 6]},
]

def option_score(cards, index):
    numerator = sum(c['weight'] * c['scores'][index] for c in cards)
    denominator = sum(c['weight'] for c in cards)
    return numerator / denominator

for i, name in enumerate('ABC'):
    print(name, round(option_score(cards, i), 2))
```

The outputs are approximately 7.89, 6.33, and 5.44. Without additional safeguards, option A wins and its score falls in the GO band.

The source defines these basic verdict bands:

- **GO:** recommended score from 7.5 to 10.
- **GO WITH CONDITIONS:** 6.0 to below 7.5.
- **RESHAPE:** 4.5 to below 6.0.
- **DON'T DO IT:** below 4.5.

It also allows a WAIT recommendation when there is a missing prerequisite, and it can apply overrides for ruin risk and irreversible decisions.

At first glance, weighted averaging seems straightforward. It becomes subtle when a score of eight means different things to different agents. One may be scoring immediate revenue. Another may be scoring the risk-adjusted business case. If those concepts are not carefully defined, the arithmetic combines incommensurate quantities.

A production-ready scoring system should document each rubric, evaluate calibration, and explicitly prevent the same evidence from being counted as seven independent observations.

**Primary source:** [Roast Room scoring specification](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/references/scoring.md).

## 8. Why the risk veto matters

Consider a decision with high apparent upside and one catastrophic failure mode. Six advisors enthusiastically recommend it. The capital specialist notices that an unfavorable outcome could exhaust the entire runway.

A majority vote would ignore the capital specialist. A weighted average might still produce a strong GO. Roast Room's specification instead applies a **ruin cap**. If any seat marks likely ruin risk on the recommended option, the overall verdict is capped at RESHAPE until an endorsed mitigation is named.

This is a form of asymmetric decision-making. Mild upside and catastrophic downside should not always cancel linearly. Some risks should be constraints rather than another number added to the score.

The source also includes a one-way-door rule. When the selected choice is difficult to reverse and advisor scores have a large spread, the recommendation cannot simply be an unconditional GO. The resolution must include a cheap test before the irreversible action.

One subtlety is often lost in descriptions of vetoes: **the risk cap changes the verdict label, not necessarily the recommended option**. Our implementation deliberately preserves this distinction. If option A is top-ranked and receives a ruin flag, the label changes from GO to RESHAPE, while A remains the highest-scoring option.

That is useful for accurately modeling the repository. It also surfaces a design issue: a UI must make the cap prominent. Otherwise, a reader might see the first-ranked option and miss the reason not to execute it immediately.

A real risk-control system should also distinguish confirmed risk from a model-generated suspicion. False-positive ruin flags can paralyze good opportunities. That is why our lab evaluates false approvals and the accuracy of the risk check separately.

## 9. The unanimity trap

The scoring document includes one of the project's most valuable rules. If every advisor selects the same best option and gives it nearly identical scores early in the process, the Chair must identify the strongest unargued alternative and require a real challenge.

Why? Because agreement can have at least three different causes.

First, everyone has reviewed strong evidence and reached the same conclusion independently.

Second, the original brief made one option look much more attractive than the others, possibly by omitting disconfirming information.

Third, the advisors are performing versions of the same learned behavior and reproducing shared assumptions.

Only the first of these is reassuring.

This connects to research on multi-agent debate. Smit and colleagues reported at ICML 2024 that debating agents do not consistently outperform cheaper self-consistency or ensembling methods. More recent work by Pitre and colleagues emphasizes that process-level metrics are necessary to diagnose sycophancy, domination, and premature convergence. Other studies find that debate can create shared errors even while agreement increases.

A practical multi-agent system should therefore record both agreement and **whether that agreement arose from independent corroboration**. The number of voices is not itself evidence strength.

**Research:** [Should We Be Going MAD? (ICML 2024)](https://proceedings.mlr.press/v235/smit24a.html), [A Diagnostic Study of Multi-Agent LLMs (ICML 2026)](https://proceedings.mlr.press/v306/pitre26a.html), [Who Convinced Whom? (2026)](https://arxiv.org/abs/2610.35930).

## 10. The local room and event-driven UI

Roast Room's visual interface is an elegant example of separating the agent execution layer from the presentation layer.

By default, the Chair writes a series of session artifacts as the process progresses. The protocol includes a decision brief, a council configuration, round-by-round Markdown cards, a list of disagreements, a draft resolution, and final verdict files.

The Node app watches these artifacts and converts them into a stream of room events. The interface can display which advisors are speaking, which arguments are being challenged, and how the final recommendation emerges.

The execution layer can therefore continue even if the visual room is closed. The room can also replay earlier sessions without rerunning any models.

The essential sequence looks like this:

```text
User's decision
    |
    v
Chair builds brief.md and council.json
    |
    v
Seven seats produce round-1.md
    |
    v
Chair writes cruxes.json
    |
    v
Debate produces round-2.md
    |
    v
Chair writes draft-resolution.md and draft.json
    |
    v
Ratification produces round-3.md
    |
    v
Chair verifies quotes and writes verdict.md
    |
    v
verdict.json marks completion
    |
    v
Local Node server broadcasts room events
```

The server binds to `127.0.0.1` on port 4730 by default, supplies endpoints for seats and sessions, and streams updates as Server-Sent Events. The default viewer mode does not itself call an AI API. Optional API mode is explicitly enabled through an environment setting and an Anthropic API key.

Architecturally, this is a lightweight event-sourced pattern, although the files are a purpose-built session record rather than a general transactional event log. It is useful because it gives developers observable intermediate state, human-readable debugging information, and replayable artifacts.

**Primary sources:** [Live-room file contract](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/references/live-room.md), [app/server.mjs](https://raw.githubusercontent.com/proxient/learn-roast-room/main/app/server.mjs), [app/lib/watch.mjs](https://raw.githubusercontent.com/proxient/learn-roast-room/main/app/lib/watch.mjs).

## 11. Provenance, quotation checking, and its limits

A simulated advisor based on a public figure creates a particular risk. The model might generate a convincing quotation that the person never said.

Roast Room tries to mitigate that by giving each seat a structured collection of source-derived material and requiring framework references. It also ships a Python script, `check_quotes.py`, that scans session text for quoted passages with source identifiers and checks whether normalized text appears in the accumulated brain material.

This is a useful integrity guard. It is not a complete citation-verification system.

The implementation searches across concatenated brain files. It can establish that a quoted sequence exists in that collected material, but does not independently retrieve the original speech or article. As written, it also does not verify that every citation identifier corresponds to the specific source containing the quotation. And its regex intentionally looks for particular quotation patterns, so it is not an exhaustive audit of every unsupported factual assertion.

This distinction generalizes far beyond Roast Room.

**Source-internal verification** asks whether a quote is present in a local collection.

**External provenance verification** asks whether that collection faithfully represents the original source.

**Causal or factual verification** asks whether the claim is true and relevant to the current decision.

These are three different problems. Passing the first does not solve the other two.

For a serious decision-support application, I would store immutable source identifiers, URLs, extraction offsets, retrieval timestamps, and hashes. I would also require a separate factual audit of market numbers and customer claims. Quoting a respected investor correctly does not prove that the investor's principle applies to the company being evaluated.

**Primary source:** [check_quotes.py](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/scripts/check_quotes.py).

## 12. Token costs and the 21-agent architecture

The repository's full-council mode performs three rounds with seven sub-agents. The README warns that this is expensive in Claude usage, and the skill file estimates about one million tokens for a full council.

The less expensive modes matter:

- `--quick`: four seats, two main rounds, with a final round only when needed.
- `--seats`: use a selected subset of specialists.
- `ask`: consult a single advisor without launching a full council.

These are not simply convenience features. They let a user choose a level of deliberation that matches the cost and irreversibility of the decision.

A weekly newsletter schedule should not require the same evaluation budget as an acquisition that could bankrupt the company. A pricing experiment can be rolled back. A term sheet, sale of equity, or irreversible market commitment may justify a larger process.

The real unit of optimization should be **quality improvement per unit of time and compute**, not simply how many agent responses were generated.

We should measure at least four costs separately: number of model calls, billed input and output tokens, elapsed time including parallelism and tool waiting, and the cost of any independent evidence gathering. The optional direct API runner records token usage, but its costs and the Claude Code skill's consumption should not be conflated.

Our benchmark uses a fictional proxy-token budget to illustrate these tradeoffs. It is deliberately not described as actual Anthropic billing.

## 13. Research on multi-agent debate

There is a substantial gap between the intuition that debate helps reasoning and experimental evidence showing when it does.

The 2024 ICML paper *Should We Be Going MAD?* benchmarks several multi-agent debate strategies against other prompting and ensemble methods. Its main conclusion is cautionary: debate systems do not reliably beat alternatives simply because multiple agents exchange arguments.

The 2026 ICML diagnostic study goes further. It argues that final agreement and the quality of a final answer are incomplete measurements. A useful evaluation should also ask whether agents engage with objections, revise appropriately, become dominated by one speaker, or converge before considering important alternatives.

The October 2026 *Who Convinced Whom?* preprint studies the origins of opinion changes in multi-agent systems. It reports that stronger consensus can coexist with more shared errors. This directly challenges the idea that consensus is an unambiguous success metric.

Other research on persuasion-driven adversarial influence shows that one compelling but incorrect participant can redirect a group. This is particularly relevant when an application explicitly instructs agents to be forceful debaters. A persuasive role may be excellent at producing disagreement but poor at distinguishing evidence from rhetoric.

None of these studies demonstrates that Roast Room itself is inaccurate. They identify **failure modes that a rigorous Roast Room evaluation should test**.

That is exactly what the companion experiment is designed to do.

---
## 14. The independent benchmark we will build

We need a controlled environment in which a council can make genuinely wrong decisions. Actual business decisions do not provide immediate ground-truth labels. A new product might fail because the market was wrong, because execution was poor, or simply because of an unrelated event. Measuring a real council fairly would require longitudinal outcomes, matched decisions, and careful treatment of confounders.

Our educational alternative is to generate synthetic decisions where we define the outcome in advance.

Each case has three possible options and seven evaluation dimensions. The underlying value of each option on each dimension is known to the evaluator but hidden from the simulated decision-makers. Each dimension has a relevance weight. The oracle chooses the option with the highest weighted true score. Some options also carry a separately generated irreversible downside flag.

Our simulated specialists do not produce natural language. Each receives a noisy observation of its assigned dimension and returns a score and risk indicators. A generalist instead receives a noisy summary of the entire option. We can then introduce shared framing errors, cross-agent averaging, an evidence audit, and the repository-inspired ruin cap one mechanism at a time.

This simplification is deliberate. It lets us measure whether a mechanism causes the claimed improvement. It does not let us infer how a language model would behave with an actual founder, public documents, or strategic uncertainty.

The experiment evaluates 600 independently generated cases for each of five seeds. That is 3,000 cases per configuration. The seven configurations are evaluated on the same cases for each seed. Some paired comparisons even reuse exactly the same underlying random observations, making the isolation of a single intervention easier to interpret.

The project runs with Python's standard library. No external model is downloaded. No private company data is used. No Anthropic API calls occur.

## 15. Generating decisions with known answers

The input generator builds a small utility matrix. A row represents an option. A column represents one of seven business dimensions. Each entry lies between zero and ten.

For realism, options share some general attractiveness. One proposal may look promising across several dimensions. But the individual specialists also see different strengths and weaknesses.

Relevance weights use values one, two, and three. Two leading dimensions get the highest weight, while some others receive intermediate weights. This mirrors the structure of Roast Room without claiming to reproduce any particular advisor's reasoning.

A simplified version looks like this:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Decision:
    values: tuple[tuple[float, ...], ...]
    weights: tuple[int, ...]
    ruin: tuple[bool, ...]

    def true_scores(self):
        total_weight = sum(self.weights)
        return [
            sum(value * weight for value, weight in zip(row, self.weights))
            / total_weight
            for row in self.values
        ]
```

The key property is that `true_scores()` belongs to the **evaluation environment**. The simulated advisors receive observations derived from this structure, not the complete ground truth.

The actual implementation also gives each case a shared-framing vector. That vector represents a compelling but potentially inaccurate narrative about an option. In the independent baseline its influence is small. In the correlated-error condition, multiple seats inherit the same misleading signal.

This means we can hold the underlying decision constant while changing how much the agents' errors correlate. A council that succeeds only when its agents make independent mistakes has a very different risk profile from one that remains robust when they all receive the same poor source material.

## 16. Simulating specialists without LLMs

Our seven specialists have no names borrowed from public figures. They represent functional dimensions: strategy, product, pricing, marketing, people, operations, and capital.

Each specialist returns three scores. The scores are noisy versions of the corresponding true dimension values. This is intentionally different from the richer natural-language cards in the real repository.

The important controls are:

- **Independent noise:** each specialist makes its own mistakes rather than copying the other agents.
- **Shared framing:** all specialists are influenced by a common misleading component.
- **Evidence audit:** measurements become less noisy because an independent process checks them.
- **Debate without evidence:** scores move toward the other agents' average, reducing differences without adding facts.
- **Risk signal:** each seat has a tunable chance of detecting a real catastrophic-risk flag and a tunable chance of falsely raising one.

Here is the heart of the specialist observation model:

```python
specialized_sigma = 1.45 / (2.5 if evidence else 1)
noise = rng.gauss(0, specialized_sigma)
framing = shared_framing[option] * (
    0.90 if correlated else 0.16
)
score = max(0.0, min(10.0, actual_dimension_value + noise + framing))
```

There is no claim that the numerical values approximate Claude's behavior. They are experimental settings chosen to expose mechanisms clearly. You can change them and observe when conclusions reverse.

The result is a **causal toy model of an orchestration policy**, not a benchmark of the original project. That distinction matters because impressive synthetic results can be made arbitrarily large by selecting favorable noise distributions. The meaningful outcome is which mechanisms are robust across changed assumptions.

## 17. Reproducing relevance-weighted aggregation

Our aggregator takes immutable seat cards and calculates a weighted score for each option. It selects the highest-scoring option, records the numerical spread between seats, assigns a verdict band, and optionally applies the ruin cap.

The main calculation is short:

```python
def score_option(cards, index):
    numerator = sum(
        card.weight * card.scores[index]
        for card in cards
    )
    return numerator / sum(card.weight for card in cards)
```

The simplicity of this function is an advantage. We can test it against hand-calculated results. The corresponding unit test creates two cards where one seat has weight three and the other weight one. It verifies that the weighted score and winning option are exactly what we expect.

The larger system also records the spread among advisors on the winning option. This is not a measure of calibrated uncertainty, but it is useful descriptive information. Two options can have the same mean score even if one has scores tightly clustered around seven while the other has some seats at three and others at ten.

It would be a mistake to convert spread directly into probability of success. Spread captures expressed disagreement, not empirical predictive confidence. If all advisors have the same bias, they may agree perfectly and still be wrong.

The lab reports spread alongside accuracy so we can test exactly that failure mode.

## 18. Implementing the ruin cap

We reproduce the repository's distinction between ranking an option and authorizing an unconditional positive verdict.

```python
label = band(winning_score)

if ruin_cap and any(
    card.ruin_flags[winning_option]
    for card in cards
):
    if label in ('GO', 'GO WITH CONDITIONS'):
        label = 'RESHAPE'
```

There are two safeguards in this tiny code fragment.

First, one specialist can raise an issue that arithmetic consensus would otherwise bury. Second, the code does not silently replace the highest-scoring option with another proposal. It changes the recommendation's permission level rather than rewriting the vote.

The downside is that the risk detector can itself make mistakes. If no seat identifies the underlying danger, the cap cannot trigger. Conversely, if a seat falsely reports a catastrophic risk, the council may delay a reasonable decision.

This is why a strong system should be evaluated with **risk-detection sensitivity and specificity**, not only the number of decisions that end with a reassuring label.

One more caveat: the generated risky flag is a hidden truth variable. The fictional advisors do not know it. They only observe probabilistic signals. The evaluator is allowed to inspect the truth afterward.

## 19. Seven benchmark configurations

The companion lab runs the same decision generator through seven policies:

**1. Solo generalist.** One noisy, holistic score for each option. It gets one simulated call and provides our cheap baseline.

**2. Four-seat quick council.** A specialist subset that mimics the source's relevance-weighted selection principle. This tests whether partial specialization provides most of the benefit at a lower budget.

**3. Seven independent specialists.** One seat for each dimension, with mostly independent measurement errors. This is the favorable setting for ensembling.

**4. Seven correlated specialists.** The same seven roles, but every seat inherits a strong shared framing error. We are testing whether specialization survives a common informational failure.

**5. Debate with no new evidence.** The same underlying observations as the correlated condition, followed by a score update toward the group average. It simulates conformity rather than the full linguistic process of argumentation. It is intentionally a negative control, not a faithful implementation of Roast Room's debate prompts.

**6. Audited specialists.** Independent verification reduces the simulated observation error and improves downside detection. We count extra model-equivalent work for the audit.

**7. Audited specialists with a ruin cap.** The exact same simulated evidence as the audited condition, but with the verdict cap enabled. Because the observations are paired, the cap's effects can be attributed to the rule rather than different random cases.

Call counts and proxy tokens are fixed synthetic budget assumptions. They are not actual messages sent to Claude and they are not measured API consumption.

The lab includes a `--seeds` flag and saves both per-seed measurements and aggregate statistics, including sample standard deviation. Do not judge a multi-agent approach on a single lucky run.

## 20. Accuracy is not enough: utility regret

A useful decision metric should recognize that some mistakes matter much more than others.

Imagine option A has true utility 7.1 and option B has 7.0. Picking B is technically an error, but the loss is tiny. Now suppose option C has true utility 3.0. Picking C instead of A is a much more consequential error.

Our experiment reports both **oracle choice accuracy** and **mean utility regret**. Regret is the difference between the best available option's true utility and the utility of the option the policy selected.

```python
scores = case.true_scores()
best_utility = max(scores)
chosen_utility = scores[OPTIONS.index(verdict.recommended)]
regret = best_utility - chosen_utility
```

This requires a known oracle, which we only have because the problem is synthetic. For a real startup decision, the appropriate analog might be realized revenue, downside-adjusted cash flow, or regret inferred from a carefully designed counterfactual analysis. Those measurements are harder and slower to obtain.

We also track the percentage of all cases in which the selected option is truly risky **and** the verdict is an unconditional GO. This measures dangerous unconditional approval under our constructed risk model.

It is not a complete safety metric. A risky option can still receive GO WITH CONDITIONS if the hazard was not flagged, and no model of downstream mitigation effectiveness is included. The experiment therefore measures a narrow property of the verdict rule, not a guarantee of responsible action.

## 21. Our measured results

We ran five deterministic seeds, each with 600 synthetic three-option decisions. The following are **averages from our runnable Python lab**. They are not results from the public Roast Room repository.

**Solo generalist** selected the best option 54.0% of the time, with mean regret 0.611. It required one modeled call and an average of 915 fictional proxy tokens per decision.

**Four-seat quick council** reached 73.6% accuracy and mean regret 0.196, at four modeled calls and 3,000 proxy tokens.

**Seven independent specialists** reached 81.7% accuracy and mean regret 0.103, at seven modeled calls and 5,085 proxy tokens.

**Seven correlated specialists** dropped to 60.2% accuracy and mean regret 0.444, despite having the same seven seats and budget.

**Debate without new evidence** produced 59.7% accuracy and mean regret 0.453, while increasing the modeled calls to 21 and the fictional token budget to 14,185.

**Audited specialists** reached 86.6% accuracy and mean regret 0.045, at a 14-call budget of 9,635 proxy tokens.

**Audited specialists plus ruin cap** retained exactly the same 86.6% option-selection accuracy and 0.045 regret, since the cap does not change the chosen option. But the proportion of cases receiving an unconditional GO on a truly risky selected option dropped from 2.9% to 0.0% under this particular simulation.

The last figure should not be read as proof that the veto eliminates risk. It means the mechanism prevented that **specific class of verdict error** in these samples. We did not simulate business execution or verify whether required mitigations would actually work.

These results make three distinct points.

First, useful specialization can substantially outperform a noisy generalist when specialists bring independent measurements of relevant dimensions.

Second, the same number of agents can perform much worse when they inherit a common mistake. Agent count did not change, but information quality did.

Third, an explicit rule for handling catastrophic downside can improve the appropriateness of the verdict without changing the winning option or claiming to improve factual knowledge.

## 22. Why correlated errors defeat multi-agent ensembles

The largest performance reversal occurs between independent and correlated specialists.

Under independent errors, the pricing advisor might overestimate option A while the operations advisor underestimates it. When the errors have different signs, aggregation can partly cancel them. A weighted average often improves the overall estimate.

Under shared framing, all seven advisors may inherit the same seductive premise: *Every competitor is raising prices, so demand must be strong.* They then use different business vocabulary to justify a common overestimate.

The group appears intellectually diverse. The errors are not.

Our simulation captures that by giving each agent an individual noise term and a shared bias. When the common component becomes strong, aggregation no longer reduces it. A larger number of seats can make the same mistake look more thoroughly examined.

The result is especially relevant to retrieval-augmented agent systems. If every specialist reads the same flawed web page, the repeated citations may create an illusion of independent corroboration. A serious council should distinguish evidence sources and mark when two claims descend from the same document.

A good next experiment would replace the arbitrary shared-bias variable with an explicit source graph. Agents would retrieve evidence nodes, and the aggregator would discount dependent testimony from the same source. This would turn the diagnosis into a concrete architecture improvement.

## 23. How debate can reduce disagreement without increasing truth

The conversation-only condition is intentionally simplistic. It takes the seven correlated scores and moves each closer to the group average. It does not model reasoning with language or reproduce the source's actual three-round protocol.

Why include it? Because it isolates one seductive failure mode: **convergence without learning**.

In the measured benchmark, mean score spread fell from about 5.65 in the correlated condition to about 2.70 after simulated convergence. That looks like a success if our metric is agreement.

But accuracy moved from 60.2% to 59.7%. Mean regret rose from 0.444 to 0.453. We paid for additional modeled discussion and became more certain-looking without becoming more correct.

The important conclusion is not that discussion is useless. It is that **agreement alone is an unsafe proxy for the value of discussion**.

A useful debate round should identify a concrete assumption and demand a discriminating test. If one advisor argues that customers will churn, the Chair could commission a churn analysis. If another worries about runway, the Chair could request an auditable cash model. The output of that tool call then becomes evidence every seat can inspect.

In that setup, additional rounds have a pathway to higher accuracy: new information enters the decision process. Without such a pathway, more rounds mainly rearrange the same claims.

This is consistent with the broader literature's recommendation to evaluate debate processes, not just final consensus scores.

## 24. Better evidence versus more discussion

The audited-specialist configuration improves observation quality rather than making agents more conversational. The simulation reduces specialist noise and improves detection of high-impact downside. It also charges a larger budget.

That is why it outperforms the conversation-only variant in the synthetic results.

This does **not** establish that a real-world financial auditor or research tool will always beat more discussion. The noise reduction is explicitly built into our simulator. The result demonstrates what happens *if* a verification step supplies genuinely more accurate evidence.

The practical question for an engineering team is therefore not *How do we get agents to debate more?* It is *Which uncertainty can a tool resolve more cheaply than additional language generation?*

For a pricing decision, the tool may be a calculator that tests customer retention and margin sensitivity. For hiring, it may be an employee capacity model. For market expansion, it may be a source-verified analysis of distribution costs. For a product pivot, it may be a small customer experiment with explicit success criteria.

The best AI agent architecture often leaves the hardest factual operations to independently checkable tools and uses language models to propose hypotheses, interpret results, and explain tradeoffs.

## 25. A worked example from the executable lab

The included `decision_lab.demo` script builds four seat cards using a fully transparent synthetic example.

Product, pricing, capital, and operations specialists score options A, B, and C. The high-weight product and pricing seats favor A. Capital notices a likely catastrophic downside for that same option.

Run:

```bash
python -m decision_lab.demo
```

The program reports:

```text
Scripted option A/B/C, synthetic cards (not real advisors)
Relevance-weighted scores:
  A: 7.89
  B: 6.33
  C: 5.44
No cap: A -> GO
With ruin cap: A -> RESHAPE
Notice: the recommended option is unchanged. The verdict label is capped.
```

This is a useful teaching example because it demonstrates the exact difference between option ranking and permission to act. The ranking function gives A the highest score. The downstream-risk rule prevents that score from automatically authorizing immediate execution.

A production system should extend this separation further. It should track which risk was detected, what evidence supports the warning, whether a mitigation is concrete and independently verified, who can approve the exception, and how the decision will be revisited.

The code also lets you run the larger benchmark:

```bash
python -m unittest discover -s tests -v

python -m decision_lab.benchmark \
  --seeds 5 \
  --cases 600 \
  --output results/benchmark.json
```

Results are saved as JSON. The core benchmark has no third-party dependencies and does not contact any external service.

## 26. Running the real Roast Room

Our offline benchmark is not the project itself. To use the actual source repository, follow the instructions in its README.

```bash
git clone https://github.com/proxient/learn-roast-room.git
cd learn-roast-room
npm start
```

Open `http://localhost:4730` in your browser. Start a separate terminal in the same directory and run:

```bash
claude
```

Then give Claude Code a well-formed decision:

```text
/roast-room I run a small developer-tools business.

Option A: hire a salesperson now.
Option B: contract a part-time salesperson for 60 days.
Option C: keep founder-led sales for three months.

Current monthly revenue is $18,000. Gross margin is 78%.
Cash available is $70,000. The founder can spend 12 hours
per week on sales. Success means reaching $30,000 monthly
revenue without losing more than three months of runway.

I lean toward Option B. Test that assumption.
```

Those numbers are hypothetical and deliberately provided as an example. A real council needs accurate data, not an invented business profile.

For a reversible decision, start with quick mode:

```text
/roast-room --quick Should I switch my onboarding demo to a prerecorded video?
```

You can request specific seats:

```text
/roast-room --seats hormozi,buffett Should I extend payment terms for enterprise customers?
```

Or consult one seat:

```text
/roast-room ask graham How would you test whether users actually want this feature?
```

An Anthropic API key is **not required for the default Claude Code workflow** if you have the required Claude Code access. The optional direct API runner is a different execution mode and can incur API billing. Check the repository and Claude Code product documentation for current plan and availability requirements before deploying.

It is worth starting by replaying the three bundled example sessions. This lets you inspect the expected format, round structure, and final report before committing a large amount of model usage.

## 27. What I would improve in the original design

The repository is already unusually explicit about role boundaries, scoring and risk controls. The next engineering improvements should focus on making the evidence harder to manipulate and the final decisions easier to audit.

**A structured fact registry.** Store every business-critical claim with a source, retrieval date, units, and verification status. Do not allow the same unverified claim to silently become a fact just because multiple seats repeat it.

**Evidence dependency tracking.** If three agents cite the same source, those are three interpretations of one source, not three independent observations. A provenance graph could reveal hidden common dependencies.

**A holdout evaluator.** The final recommendation should be checked against constraints that the debating agents cannot rewrite. For example, a deterministic evaluator could check that cash never falls below a required reserve under the supplied scenario assumptions.

**Calibration against real outcomes.** Track decisions and their results over time. Evaluate where specific seats provide useful incremental predictive information, and learn weights from properly separated historical data instead of treating intuitive weights as immutable truths.

**Cost-aware adaptive routing.** Use a single seat for simple, reversible questions. Add specialists when the expected value of information justifies it. Trigger a third round only when important uncertainty remains and there is a credible route to resolving it.

**A structured dissent report.** The strongest minority objection should not be reduced to decorative prose. It should have its own assumptions, falsification test, and criteria for changing the final call.

**More robust provenance enforcement.** Quote checking should validate the cited source identifier and original passage, not merely whether similar words appear somewhere in a combined brain folder.

**Separation of roles and authority.** A high-confidence recommendation should remain advice. The system should never silently convert a simulated advisor's verdict into a financial transfer, customer notification, or contract execution.

## 28. Threat model and practical deployment checklist

This application processes information that founders may consider commercially sensitive. A pricing experiment, acquisition proposal, or financing decision can contain revenue numbers, customer information, private strategy, and personnel details.

The default local room binds to loopback, which reduces accidental network exposure. But the council still writes session artifacts to disk, and those artifacts may contain the brief and detailed analysis. Review file permissions, source-control exclusions, backups, retention, and workspace sharing before using real company information.

There are also language-model-specific risks.

**Prompt injection through context.** If a brain document, retrieved web page, or business note contains instructions that masquerade as part of the orchestration protocol, an agent may follow them. Treat retrieved information as evidence, not authority.

**Source authority confusion.** An apparently authentic quote can be irrelevant to the current situation. Correct attribution is necessary but insufficient.

**Repeated evidence.** Several agents may cite the same unverified market estimate. Do not count those as independent validations.

**Confident numerical fabrication.** A model can produce a polished cash-flow argument with wrong arithmetic. Run actual calculations through deterministic tools.

**Risk-cap bypass by omission.** A veto rule does nothing if no seat knows the relevant danger exists. Perform an explicit constraint and downside inventory before voting.

**Decision laundering.** An attractive room animation and PDF can make speculative recommendations look institutional. Make uncertainty, unresolved assumptions, and minority objections visible in the primary view.

For higher-stakes use, the safest design is human review supported by deterministic verification and preserved evidence. The AI council can assist judgment without substituting for legal, financial, or professional accountability.

## 29. Limitations and experimental honesty

The original Roast Room is a Claude Code system using natural-language roles, large context windows, retrieved source material, and an interactive Chair. Our companion lab is a synthetic statistical simulator using ordinary Python.

The differences are substantial:

- The simulated agents do not read or reason about business documents.
- Their errors are generated from mathematical distributions rather than neural network behavior.
- Their role specialization is explicitly defined by which dimension they observe.
- The evidence audit directly reduces observation noise by design.
- The conversation-only ablation models score convergence, not genuine argument exchange.
- The fictional token budget is not an API-cost estimate.
- The oracle utility function has no equivalent guarantee in actual business outcomes.
- The ruin cap is tested on synthetic risk flags, not real-world irreversible harm.

The experiments consequently cannot demonstrate that Claude Code with seven agents is more accurate than one Claude instance, that the repository provides a 27-point accuracy improvement, or that three rounds of its actual debate degrade results. Any such statement would require genuine controlled model executions and independent labels.

The lab **does** give a fully inspectable way to test theoretical properties of agent councils: independent evidence is beneficial under its assumptions; shared bias can negate that benefit; agreement can increase without truth increasing; and a verdict cap can change authorization without changing rankings.

That is already valuable, provided the boundaries remain clear.

## 30. Final lessons and further experiments

Roast Room is compelling because it does not treat multi-agent orchestration as merely spawning several chat windows. Its seats have different responsibilities. Its Chair must explain how it combines scores. Its workflow preserves dissent and intermediate reasoning. It takes irreversible risks seriously. And its local app provides a replayable view of the process.

The counterargument is equally important: specialized personas are not independent evidence. An extra debate round is not a new measurement. A weighted average does not automatically handle tail risks. A source note is not the same as external verification.

Our results suggest a useful order of operations for anyone building decision agents.

Start with a strong single-agent baseline. Introduce specialists only when they bring distinct information or tools. Add disagreement and provenance measurements so consensus cannot hide shared mistakes. Use deterministic checks for arithmetic, contracts, and operating constraints. Increase the number of rounds only when new evidence can change the answer. Treat strong verdicts as conditional recommendations, not permissions to act.

From there, several research extensions become possible. One is a real Claude Code evaluation using matched budgets and human-scored business cases. Another is a source-provenance graph that discounts dependent citations. A third is a learned router that decides which specialists to summon from expected value of information. A fourth is a calibrated risk-veto model evaluated on adversarial cases. A fifth is a disagreement-aware stopping criterion that terminates debate when new rounds stop producing verifiable facts.

The real opportunity is not to build an AI board that sounds like successful founders.

It is to build one that makes **fewer expensive mistakes**, explains what evidence it used, and knows when a decision still requires a human.

---

## 31. References and source links

**Primary repository sources**

1. [Roast Room, linked fork by `proxient`](https://github.com/proxient/learn-roast-room). Repository landing page and code.
2. [Original Roast Room by `nateherkai`](https://github.com/nateherkai/roast-room). Upstream attribution.
3. [Roast Room skill protocol](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/SKILL.md). Mode selection, Chair duties, orchestration, and council rounds.
4. [Seat definitions and relevance weighting](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/references/seats.md). Specialization and selection criteria.
5. [Round prompts](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/references/round-prompts.md). Structured advisor outputs and debate procedures.
6. [Scoring rules](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/references/scoring.md). Weighted averages, verdict bands, risk and one-way-door overrides.
7. [Live-room data contract](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/references/live-room.md). Session files and viewer semantics.
8. [Local Node server](https://raw.githubusercontent.com/proxient/learn-roast-room/main/app/server.mjs). Viewer mode, optional API mode, and streaming endpoints.
9. [Quote-checker implementation](https://raw.githubusercontent.com/proxient/learn-roast-room/main/.claude/skills/roast-room/scripts/check_quotes.py). Internal source-note verification and its scope.

**Research and platform documentation**

10. Anthropic. [Steering Claude Code: Skills, hooks, and subagents](https://claude.com/resources/articles/steering-claude-code-skills-hooks-rules-subagents-and-more). Explanation of sub-agent isolation and tool boundaries.
11. Smit et al. [Should We Be Going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://proceedings.mlr.press/v235/smit24a.html). ICML 2024.
12. Pitre et al. [A Diagnostic Study of Multi-Agent LLMs for Real-World Debates](https://proceedings.mlr.press/v306/pitre26a.html). ICML 2026.
13. Batra et al. [Who Convinced Whom? Tracing How LLMs Change Their Minds in Multi-Agent Systems](https://arxiv.org/abs/2610.35930). October 2026 preprint.
14. [When Collaboration Fails: Persuasion-Driven Adversarial Influence in Multi-Agent Large Language Model Debate](https://www.nature.com/articles/s41598-026-42705-7). Scientific Reports 2026.

**Companion experiment:** See the `roast_room_decision_lab` Python project for source code, correctness tests, seeded results, experimental methodology, and figures. The benchmark is an independent educational implementation and is not affiliated with the Roast Room maintainers.
