# Research references and interpretation

- Mohammed Nayeem Khan and Logesh L, *Maya a1: Do the Options Read the Model, or Does Their Position?*, arXiv:2611.09803v1, November 6, 2026: https://arxiv.org/abs/2611.09803
- Laya project and official parallel option layout: https://github.com/NandhaKishorM/laya
- Model weights: https://huggingface.co/convaiinnovations/laya
- Pezeshkpour and Hruschka (NAACL 2024): https://aclanthology.org/2024.findings-naacl.130/
- Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors*: https://arxiv.org/abs/2309.03882
- Warner et al., *ModernBERT*: https://arxiv.org/abs/2412.13663

Paper findings **as authors report**, not reproduced here:

- 951 Qwen-family-filtered multiple-choice questions; 3/5-option full cyclic rotations and six sampled permutations for 12 options.
- Base checkpoint flips on 25.8% of questions, with 19.7% on English-only subset (n=605).
- No-shuffle fine-tuning: 11.4% flip. Matched shuffled fine-tuning: 7.2%. Parallel layout trained on the same data: 0.5%.
- Flip rate is **per-question answer instability** across sampled presentation orders, not percentage of individual inference calls wrong.
- Single-pass accuracy 69.1% base and 86.3% shuffled-fine-tuned on teacher-filtered data. These are not human-labeled gold benchmarks.
- Confidence-weighted vote yields 69.7% for base at 4.1x average inference compute; authors' own infrastructure may batch orderings, so wall-clock latency is not necessarily 4.1x.
- Authors' parallel layout existed in Laya code and was not a newly invented method by Maya researchers. The Maya paper experimentally fine-tunes and evaluates it on their data.
