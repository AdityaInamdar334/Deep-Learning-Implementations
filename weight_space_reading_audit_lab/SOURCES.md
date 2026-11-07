# Sources and distinctions (verified November 7, 2026)

1. Nazmus Sakib Ahmed, *Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring "Weight-Space Programming"*, arXiv:2611.08070v1 (Nov 5 2026): https://arxiv.org/html/2611.08070v1
   - Single model (Claude Opus 5.5), mostly single runs, small trained CNNs with ~125k parameters, no tools, no forward passes.
   - 99.57%-accurate SAT-6 network interpreted as CIFAR-10 animals. Output permutation retained structural split but broke semantic label assignments.
   - Correlations compared to independently computed data values; correct estimates do NOT establish unique semantic recovery from weights.
   - No direct forward-pass tracing validation; author says conclusions are exploratory and not a refutation of editing experiments.
2. Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, *Sparks of Weight-Space Programming*, Nov 2 2026: https://rikiyat.github.io/blog/weight-space-programming/
   - Reports separate experiments on reading, writing, and editing. GPT-6 Astra editing Qwen3.5-0.8B uses Python and multiple model forward evaluations, but no gradient descent/training loop. Authors report MATH-500 7.9% -> 48.1%.
3. Simon Willison, *Qwen 3.8 27B on my laptop can interpret model weights, too*, Nov 3 2026: https://simonwillison.net/2026/Nov/3/qwen-can-interpret-model-weights/
   - Personal no-tool reproduction of 3 out of 4 game-identification examples, over tens of minutes each; not a statistical replication or a test on Ahmed's CNN prompts.
4. Matthew Honnibal, *The 'Astra can read and write weights to improve Qwen' post is misleading*: https://honnibal.dev/blog/astra-read-write-misleading
5. Sean Goedecke, *LLMs are not superhuman at reading and writing neural network weights*: https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/
6. Independent OpenWSP reproduction and tooling: https://github.com/openwsp/openwsp (maintainer reports a 12.9% MATH-500 live edit as of check, below paper's 48.1%; reproduction conditions differ; do not assume this negates original result)

**This lab's output is independent synthetic evidence for a method, not a replication of any source's empirical model measurements.**
