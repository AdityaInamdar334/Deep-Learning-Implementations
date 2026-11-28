# Optional replacement for Section 8 (new executed experiment)

The original article describes an earlier synthetic study whose exact executable source is no longer available. This repository independently implements the same research question, with a different synthetic generator and three training seeds. If publishing the article with this repository, replace Section 8's numerical findings with the following:

I generated 6,000 synthetic decision states for each of three seeds, with four candidate answers per state. I deliberately biased the canonical ordering toward the correct answer appearing first. I trained three scoring models and evaluated them across every one of the 24 possible candidate permutations on 1,000 held-out states per seed.

The position-biased scorer achieved 93.30% canonical accuracy, but only 74.61% accuracy averaged across permutations. Its selected answer changed on 53.27% of questions. Shuffling candidate order during training reduced the flip rate to 2.47%, while the shared, position-free scorer produced no flips in the tested states and achieved 82.67% accuracy across permutations.

[Insert figures/permutation.png and figures/accuracy.png]

These are controlled synthetic results, not evaluations of Qwen, Jev or MAYA. The exact numbers differ from the earlier article experiment because this is a new implementation. The underlying lesson remains: test the same decision under different candidate orders.

Also update Section 17 to cite the new calibration results in results/offline.json, or explicitly identify the 72% coverage / 97.92% accuracy numbers as the earlier experiment, not the new repository. Section 5's 72% token-budget example remains an illustrative assumption. The new support demo deferred all held-out cases, so do not describe it as achieving measured token savings. The optional Qwen training code remains unexecuted.
