# Research references (verified Nov 7, 2026)

1. Google Open Source Blog announcement, Oct 22, 2026: https://opensource.googleblog.com/2026/10/introducing-elephant-goldfish-model-plugin.html
2. Official EGM repo and README: https://github.com/google/elephant-goldfish-model
3. Official `egm-review` protocol: https://github.com/google/elephant-goldfish-model/blob/main/skills/egm-review/SKILL.md
4. Official harness capability and isolation caveats: https://github.com/google/elephant-goldfish-model/blob/main/docs/harnesses.md
5. EGM testing playbook: https://github.com/google/elephant-goldfish-model/blob/main/TESTING.md
6. First v0.1.0 prerelease announcement: https://github.com/google/elephant-goldfish-model/discussions/184
7. Dave Rensin's essay, April 27, 2026: https://drensin.medium.com/elephants-goldfish-and-the-new-golden-age-of-software-engineering-c33641a48874
8. GitHub Spec Kit official documentation: https://github.com/github/spec-kit/blob/main/docs/reference/agentic-sdd.md
9. OpenSpec official workflows: https://github.com/Fission-AI/OpenSpec/blob/main/docs/getting-started.md

**Scope note:** Current repository has `scripts/` and `tests/`; the core review behavior remains agent-driven and should not be confused with an independently enforced approval system. Google's internal-use figures are vendor claims, not independent effectiveness studies. Real fresh subagents depend on the host, mode, and permissions. There is no dedicated default security audit in the documented five-stage review, so the lab adds one intentionally.
