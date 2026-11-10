# Matt Pocock’s 12-Line AI Skill Grew to 30 Lines. What Changed?

*Inside Skills for Real Engineers, where a small change to how a coding agent asks questions can reach all the way to the test that catches a production bug.*

Give a coding agent this ticket: **“Let users cancel a training run.”**

It can add a Cancel button, wire up an endpoint, turn the status badge gray, and write a passing test for the whole interaction. Meanwhile, a worker can still pick up the cancelled job and spend the next six hours training a model nobody wants.

Before anyone writes the button, someone has to ask a question whose answer changes the implementation. **Does cancellation mean the user no longer wants the result, or have we promised that a queued run will never start?**

That is the kind of decision Matt Pocock’s `grilling` skill tries to surface. An early version fit in twelve lines of Markdown. By v1.3.2 it occupied thirty. I checked both files. In July the agent was told to ask one question, wait, and then ask the next. By October it was instructed to identify which decisions were independent, ask those together, and revisit the dependent ones after the user answered. An engineer reading the diff can see a specific tradeoff being worked out: how to get enough human judgment without making the human sit through a needlessly serial interview. [1, 2]

The numbers that made the repository famous need their own correction. On November 10, 2026, skills.sh displayed **32.4 million total installs across the collection**, approximately **1.1 million for `grilling`**, and **1.5 million for `grill-me`**. The familiar “20 million installs for a twelve-line file” headline combines a collection-wide number with one file from an earlier release. These counters tell us about distribution. They give us no count of distinct engineers or evidence of fewer bugs. [3]

## Contents

1. What changed inside the interview
2. How one product decision becomes a testable spec
3. Who gets to start the next stage
4. The regression test that earns its red
5. What setup puts in your repository
6. What the releases and install counts actually say
7. Where this approach fits a team
8. Sources and a reproducible audit

## 1. What changed inside the interview

Pocock, the creator of Total TypeScript, publishes the skills he uses in his own work under the MIT license. His README organizes the motivation around four recurring experiences: the agent built the wrong thing, its language is cumbersome, its code fails, or the codebase becomes hard to change. The folders themselves are organized into categories such as engineering and productivity. The four failures are a way to *read* the collection rather than its directory structure. [1, 4]

The interview illustrates why the distinction matters. There are three names a new reader can easily confuse. In v1.3.2, `/grill-me` is a user-facing entry point whose body delegates to `grilling`. The `/grill-with-docs` entry point delegates to two skills, `grilling` and `domain-modeling`. The reusable interview lives in `skills/productivity/grilling/SKILL.md`. Counting the wrapper tells you almost nothing about what happens in the conversation. [1, 2]

The v1.1.0 primitive is recognizably an interview you could use with another engineer. Resolve decisions in dependency order. Supply a recommended answer. Look up facts that are already available in the code. Give the human time to answer. Confirm shared understanding before carrying out the plan. Its twelve lines include the frontmatter and blank lines. [2]

The October version changes what happens when three questions are ready and none depends on the others. It asks them in the same round. Each gets a recommendation, phrased so a simple yes accepts it. Information the agent can obtain from the environment is assigned to exploration. Questions downstream of that research wait, while unrelated decisions can proceed. The developer still decides what behavior the software should promise. [2]

For the training-run feature, imagine the first round containing these two questions:

1. **Which runs can a user cancel?** I’d start with queued runs. Stopping a running process introduces a separate shutdown contract.
2. **Who may request cancellation?** I’d use the same ownership check as the existing run-management endpoint.

Either answer can change independently. Asking whether to send a termination signal is premature until someone decides that cancelling a *running* run is in scope. Asking the user which table stores jobs is wasteful if the agent can read the worker. There is also a question the user may have no reason to know: can two workers claim the same run? That belongs in an inspection of the claim operation.

The recommendations are part of the design. They make an interview cheap when the defaults are sensible. They also create an anchoring effect: a busy reviewer can say yes to a plausible proposal without realizing that it defines a customer-visible guarantee. A good question makes the consequence audible. “Queued runs only” is easier to evaluate when followed by “a running job will continue consuming GPU time.”

*The training queue is an original worked example throughout this article. It is separate from the repository’s examples and from any measured claim about an AI model.*

## 2. How one product decision becomes a testable spec

Suppose the developer agrees to queued-only cancellation. Three different components now need to mean the same thing by *cancelled*. The UI displays it. The API records it. The worker must respect it when deciding whether to start a job.

A glossary entry can be short enough to earn its keep:

> **Cancelled run.** A run withdrawn while queued. A worker can no longer claim it. Repeating the cancellation request leaves it cancelled.

The last sentence resolves a real implementation question. A browser can retry a request after a timeout. If the first cancellation succeeded and its response was lost, a second click should produce a predictable result.

This gives some substance to Pocock’s second failure mode, verbosity. His README uses *materialization cascade* as a domain term in a course-management application. What makes a term pay off is the behavior the team can recover from it. If *cancelled* means “hidden in my UI” to one developer and “ineligible to start” to another, shortening the conversation makes the misunderstanding cheaper to repeat. A glossary earns its place when a reader can use it to settle a disputed example. [1, 5]

Pocock also suggests that shared vocabulary can reduce reasoning tokens. The repository illustrates shorter wording. It supplies no controlled token benchmark for that claim. Loading a glossary itself consumes context. To evaluate the tradeoff, run the same change requests with and without the relevant glossary entries, measure total usage, and grade whether the implementation preserved the intended behavior. A shorter reply is a different measurement.

Once the decision is settled, `/to-spec` turns the conversation into a document on the configured tracker. Its implementation says to synthesize the discussion and check the proposed testing seams with the user. The README elsewhere loosely describes it as quizzing the user about modules. Reading the skill itself resolves the apparent conflict: synthesis can include a focused check on *where to observe the feature* without restarting the requirements interview. [6]

For the cancellation feature, the valuable sentence in the spec is this:

> If cancellation succeeds before a worker claims a queued run, a subsequent attempt to claim that run must fail.

A spec containing that sentence gives the next engineer room to choose SQL, a queue primitive, or an in-memory transition for a prototype. It also rules out an entire family of implementations that look finished in a screenshot. This is how an interview survives a context reset. The next agent can read the invariant and discover whether its proposed change satisfies it.

## 3. Who gets to start the next stage

Pocock distinguishes user-invoked skills from skills a model can select when the task fits. His user-facing interview entry points carry `disable-model-invocation: true` and call the shared discipline explicitly. The repository’s stated composition rule keeps a user-invoked skill from calling another user-invoked skill. It can still call model-invoked skills. This is a design convention with host-dependent invocation behavior. Markdown alone cannot enforce a security boundary across agent runtimes. [1, 2]

A developer might call `/grill-with-docs`, then `/to-spec`, then `/to-tickets`. Each transition has a concrete reason. The interview resolves meaning. The spec preserves the agreement. The tickets divide the work into pieces with visible outcomes and explicit blocking relationships. A small change may need only the first conversation and a test. [6]

For the queue, one vertical ticket could deliver cancellation through the public service and verify that a subsequent worker claim is rejected. Another could expose that existing behavior in the UI. The first ticket produces an independently checkable change. A ticket for “add a status column” leaves a reviewer unable to tell whether cancellation works.

Review deserves a closer look here. The `/implement` skill requests TDD at agreed seams, runs checks, invokes `code-review`, then tells the agent to commit. Review can occur *inside* the operation the developer selected. The separate `/implement-spec` skill goes further. Its instructions cover ticket dependencies, concurrent implementation in worktrees, an integration branch, and final review. Selecting a small entry point and selecting the whole-spec orchestrator authorize very different scopes of work in the repository’s design. [7]

The same distinction explains some outdated tutorials. `/to-prd` and `/to-issues` belong to earlier repository history. A July change consolidated planning around `/to-spec` and `/to-tickets`. An old name appearing on an install leaderboard is evidence of that name’s distribution history. The plugin manifest for the release you installed determines which names that plugin ships. [3, 8]

## 4. The regression test that earns its red

Return to the gray Cancel button and the worker that still starts training. A UI test can pass while the worker is wrong. The first useful test crosses the seam between the two operations.

```python
run = Run()                    # Initially queued
assert run.cancel() == "cancelled"
assert run.cancel() == "cancelled"
assert not run.claim()         # The worker tries to start it
```

The companion implements this in plain Python. It also checks the other ordering: when the worker claims the run first, a later cancellation leaves the run running. Together the two examples specify the ordering contract. In a distributed service, making those outcomes hold for concurrent requests requires an atomic transition at the shared store. This in-memory exercise verifies the sequential behavior on either side of that boundary.

Pocock’s `/tdd` encourages building behavior incrementally through an interface. His `/diagnosing-bugs` gives a failing system a more specific sequence: reproduce the reported symptom, minimize it, form a hypothesis with a testable prediction, instrument the relevant path, fix it, and preserve the regression. In v1.3.2, phase five acquired an easy-to-miss check. When deliberately mutating code or a fixture to force a failure, compare the edit with a pristine copy and establish that the intended mutation actually landed. [9]

Here is what that check looks like in the queue exercise. The good worker rejects every run whose state differs from `queued`. The mutant rejects only runs already marked `running`:

```diff
-        if self.state != "queued":
+        if self.state == "running":
             return False
```

The mutant happily claims a cancelled run. The exercise prints the diff, verifies that the original passes, and checks that the mutant fails on the specific “worker claimed a cancelled run” assertion. You can try a second experiment: remove that assertion. The exercise then reports that the mutant escaped. You have discovered exactly which part of the contract the remaining tests leave exposed.

A diff proves the textual edit landed. A failure at the intended assertion establishes that this test detected its behavioral consequence. Both observations matter. A test suite that was already failing on a missing dependency could give you a red screen after almost any edit.

This is the failure mode behind the changelog entry. It also explains why the source repository can be worth studying even for a team that uses a different agent. The debugging discipline translates directly into something a reviewer can run. [9]

### Where the queue’s design starts to fray

A month later, a timeout handler, a worker and two API endpoints all have their own lists of states from which a run may transition. The next person to add a state has five places to inspect. This is the fourth failure in the README made concrete. Pocock’s architecture skill looks for opportunities to put substantial behavior behind a small interface. It uses recent changes to help choose where to look, presents candidates in an HTML report, and interviews the developer about the candidate they select. [14]

For our queue, a candidate would be a run-lifecycle module that owns the permitted transitions and exposes claim and cancellation operations with explicit outcomes. Ask what a caller can stop knowing after that refactor. If every caller still checks three statuses and coordinates its own transaction, the new module has mostly moved the code. If callers can rely on one operation to perform the transition atomically, the module has taken a difficult obligation off their hands. That is a design decision the team can evaluate against a real change. An architecture report by itself cannot make it happen.

## 5. What setup puts in your repository

Version 1.3.2 offers two broad distribution choices: a host plugin and editable skill files. The tagged README documents installation for Claude Code, Codex, GitHub Copilot, Gemini CLI and other supported agents. For the editable route, it gives this command. [1]

```bash
npx skills@latest add mattpocock/skills
```

The installer lets you choose skills and a target agent. The README advises choosing one installation route per agent to avoid duplicates. The skills.sh route copies editable files and has an explicit update command. Plugin update behavior depends on the host. If your aim is to inspect exactly the release discussed here, use the tagged checkout in the companion instead of relying on whatever a default install fetches today. [1]

Once invoked, `/setup-matt-pocock-skills` inspects the repository before proposing configuration. The three topics are the issue tracker, triage vocabulary and domain-document layout. These are **three possible decisions**, not a promise to ask exactly three questions. Setup skips triage labels when the triage skill is absent. It normally selects the single-context documentation layout directly and asks about alternatives when the project looks like a monorepo. [10]

GitHub, GitLab and local Markdown have defined paths. A team using Linear or Jira can describe its workflow for setup to record. That distinction matters to anyone expecting a ready-made tracker integration. The outputs include `docs/agents/issue-tracker.md`, `docs/agents/domain.md`, an optional triage-label file, and pointers in a root agent-instruction file. Another skill can then find out where to publish a spec, or which label means a ticket is ready for autonomous implementation. [10]

A detail from the October release makes the cost of bad setup concrete. The GitHub template was edited to have agents verify the target repository before creating an issue. The changelog says agents working in clones or forks had filed issues upstream. A clear ticket-writing prompt can still perform the correct operation on the wrong repository. That destination belongs in the project configuration. [9]

## 6. What changed between the releases

I counted physical lines in the tagged `grilling/SKILL.md`, including frontmatter, blank lines and the example question format. I also counted the skill entries in each tag’s Claude plugin manifest. These two commands reproduce the October counts from a Git clone:

```bash
git show v1.3.2:skills/productivity/grilling/SKILL.md | wc -l
git show v1.3.2:.claude-plugin/plugin.json | python -c \
  'import json,sys; print(len(json.load(sys.stdin)["skills"]))'
```

**v1.0.0** had 10 lines in the interview primitive and 17 plugin entries. **v1.1.0** had 12 and 21. **v1.2.0** had 22 and 25. **v1.3.0** had 28 and 27. **v1.3.2** had 30 and 27. In the last tag, counting every `SKILL.md` under `skills/` yields **38**: 20 engineering, seven productivity, seven in-progress and four miscellaneous. The manifest ships the 27 engineering and productivity entries. The skills.sh collection page displayed 56 listings on the date checked. Each count has a different denominator. [1, 2, 3]

The v1.3.0 tag dates to October 4. That release promoted `implement-spec`, added `pr` and `retro`, and changed the project vocabulary convention from `CONTEXT.md` and `CONTEXT-MAP.md` to `GLOSSARY.md` and `GLOSSARY-MAP.md`. Version 1.3.2, tagged October 29, includes the mutation check, strengthens the standards review, and directs `ask-matt` to read a skill’s actual instructions before describing it. `ask-matt` lives in engineering. The same release documents changes to `chief-of-staff`, which lives under in-progress and is absent from the 27-entry manifest. A changelog entry can describe work outside the shipped plugin. [8, 9]

A few skills are especially relevant to people who lose a long conversation midway through a project. `/handoff` writes a focused handoff in the operating system’s temporary directory and points the next agent to durable artifacts. `/teach` uses a working directory to support learning across sessions. And the promoted writing reference in v1.3.2 is `writing-for-agents`, the successor to the older `writing-great-skills` name. These file locations are more reliable guides to a particular release than a tutorial that mixes several snapshots. [1, 11]

## 7. Where this approach fits a team

The comparison with Superpowers is often presented as twelve lines versus hundreds. But `grilling` is an interview procedure, while Superpowers’ `writing-skills` describes how to author and test skills. Measuring their lengths compares different jobs. A fairer comparison is to give both workflows the cancellation ticket and see when each surfaces the race between the user and the worker. Superpowers explicitly documents a broader development methodology. GitHub’s Spec Kit likewise provides tooling for spec-driven work. Pocock’s criticism of process-heavy approaches is his design position. His own `implement-spec` shows how substantial a workflow can become when the developer elects to use it. [1, 7, 12]

The same care applies to adoption claims. GitHub displayed roughly **319.9k stars** when checked on November 10. That figure measures repository stars. The 32.4 million figure belongs to a separate service and counts installs across listed skills. Neither tells us how often a team used an installed skill, kept it enabled, or shipped a better change because of it. An author-led project can have outside contributors, too. A claim about its precise bus factor needs evidence about who can maintain and release it. [3, 4]

Thoughtworks placed the collection in **Trial** on November 4, 2026. Its entry reports value in requirements analysis and applying engineering practices while cautioning teams to evaluate individual skills before adopting a whole library. That is a practitioner assessment. The repository and Radar entry provide no controlled estimate of time saved, bugs prevented or model-token reduction. [13]

For a team considering a trial, I would start with a ticket whose wrong implementation you can describe in one sentence. For the training queue, that sentence is “the UI says cancelled, but the GPU job still starts.” Give one fresh session the ordinary ticket and another the same ticket with the interview workflow, using the same model, starting code and available tools. Have a reviewer inspect whether each session uncovers the worker-claim boundary. Record how many human interruptions it takes, whether the resulting spec defines both orderings, and whether the tests catch the mutant. Repeat across several tickets before drawing a conclusion about the workflow.

The answer might be that the extra round of questions pays for itself immediately. It might be that your team’s existing ticket template already gets the agent there. Either outcome tells you more than a leaderboard. When a user presses Cancel tomorrow, the fact you need to know is whether the worker can still claim the run. A thirty-line interview earns its place if it helps your team discover that question while the answer is still cheap to change.

## Sources and companion exercise

Source inspection uses **v1.3.2**, commit `d1478fcb140db0b93c0c8df17bcfd311844749d0`, except for explicit historical comparisons. Directory metrics were checked November 10, 2026. The companion includes a Git audit and the runnable Python cancellation exercise with captured outputs.

1. Matt Pocock, tagged README, installation instructions and manifest. [README](https://github.com/mattpocock/skills/blob/v1.3.2/README.md) · [Plugin manifest](https://github.com/mattpocock/skills/blob/v1.3.2/.claude-plugin/plugin.json) · [MIT license](https://github.com/mattpocock/skills/blob/v1.3.2/LICENSE)
2. The two interview implementations and entry points. [July primitive](https://github.com/mattpocock/skills/blob/v1.1.0/skills/productivity/grilling/SKILL.md) · [October primitive](https://github.com/mattpocock/skills/blob/v1.3.2/skills/productivity/grilling/SKILL.md) · [grill-me](https://github.com/mattpocock/skills/blob/v1.3.2/skills/productivity/grill-me/SKILL.md) · [grill-with-docs](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/grill-with-docs/SKILL.md)
3. Live distribution counters. [skills.sh collection and individual listings](https://skills.sh/mattpocock/skills)
4. Project popularity and author background. [GitHub repository](https://github.com/mattpocock/skills) · [Gergely Orosz’s interview with Matt Pocock](https://newsletter.pragmaticengineer.com/p/ai-skills-with-matt-pocock)
5. Domain modeling and the author's example vocabulary. [Skill instructions](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/domain-modeling/SKILL.md) · [Course video manager glossary at the cited revision](https://github.com/mattpocock/course-video-manager/blob/076a5a7a182db0fe1e62971dd7a68bcadf010f1c/CONTEXT.md)
6. Planning and decomposition. [to-spec](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/to-spec/SKILL.md) · [to-tickets](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/to-tickets/SKILL.md)
7. Implementation scope. [implement](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/implement/SKILL.md) · [implement-spec](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/implement-spec/SKILL.md)
8. Historical changes. [Planning consolidation](https://github.com/mattpocock/skills/commit/386d4ff) · [v1.3.0 release](https://github.com/mattpocock/skills/releases/tag/v1.3.0)
9. Testing and release changes. [Diagnostic procedure](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/diagnosing-bugs/SKILL.md) · [TDD](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/tdd/SKILL.md) · [v1.3.2 changelog](https://github.com/mattpocock/skills/blob/v1.3.2/CHANGELOG.md)
10. Configuration. [Original setup instructions](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/setup-matt-pocock-skills/SKILL.md)
11. Productivity skills. [Tagged directory](https://github.com/mattpocock/skills/tree/v1.3.2/skills/productivity)
12. The alternatives in their authors’ terms. [Superpowers](https://github.com/obra/superpowers) · [GitHub Spec Kit](https://github.com/github/spec-kit)
13. Thoughtworks Technology Radar, November 4, 2026. [mattpocock/skills](https://www.thoughtworks.com/radar/tools/mattpocock-skills)
14. Architecture survey. [improve-codebase-architecture](https://github.com/mattpocock/skills/blob/v1.3.2/skills/engineering/improve-codebase-architecture/SKILL.md)
