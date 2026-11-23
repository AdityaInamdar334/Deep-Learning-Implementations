"""Educational toy examples, NOT a reproduction of Fan et al. (2026)."""
import argparse
import numpy as np


def projection_edit(W, direction, alpha):
    """W <- W - alpha * u(u.T W), with u normalized."""
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    return W - alpha * np.outer(u, u @ W)


def paired_evaluation(base, edited):
    """Clustered by problem: each row contains repeated-seed binary grades."""
    base = np.asarray(base, dtype=float)
    edited = np.asarray(edited, dtype=float)
    assert base.shape == edited.shape and base.ndim == 2
    differences = (edited - base).mean(axis=1)
    mean = differences.mean()
    se = differences.std(ddof=1) / np.sqrt(len(differences))
    return float(base.mean()), float(edited.mean()), float(mean), float(se)


def demonstrate_projection():
    rng = np.random.default_rng(14)
    hidden_dim, outputs = 8, 5
    W = rng.normal(size=(hidden_dim, outputs))
    direction = rng.normal(size=hidden_dim)
    u = direction / np.linalg.norm(direction)
    alpha = 0.05
    updated = projection_edit(W, direction, alpha)
    before = u @ W
    after = u @ updated
    assert np.allclose(after, (1 - alpha) * before)
    assert np.linalg.matrix_rank(updated - W, tol=1e-9) == 1
    print('TOY PROJECTION EDIT')
    print('Change rank:', np.linalg.matrix_rank(updated - W, tol=1e-9))
    print('Relative norm of direction output after edit:', round(np.linalg.norm(after) / np.linalg.norm(before), 4))
    print('Explanation: this weakens one measured direction by 5% exactly.\n')


def demonstrate_paired_scores():
    rng = np.random.default_rng(26)
    n_problems, n_seeds = 100, 8
    # Hypothetical completion probabilities. Not derived from the paper.
    difficulty = rng.uniform(0.2, 0.85, size=(n_problems, 1))
    common_random = rng.random((n_problems, n_seeds))
    base = common_random < difficulty
    # Example intervention helps only a subset of difficulty profiles.
    edited = common_random < np.minimum(difficulty + 0.06, 1.0)
    b, e, delta, se = paired_evaluation(base, edited)
    print('TOY PAIRED EVALUATION (identical problems and random draws)')
    print(f'Base: {b:.3f}; edited: {e:.3f}; change: {delta:+.3f}; SE across problems: {se:.3f}')
    print('This is synthetic data, not the Qwen benchmark.\n')


def demonstrate_token_budget():
    # An exact arithmetic illustration. Not a model benchmark.
    tokens = [800, 1300, 2600, 5200, 8100, 12000, 18000]
    print('TOY TOKEN-BUDGET EXAMPLE')
    for budget in [4096, 16384]:
        cutoffs = sum(n > budget for n in tokens)
        print(f'Budget {budget:5d}: {cutoffs}/{len(tokens)} responses truncated')
    print('More tokens can reduce truncation but increase inference cost.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', choices=['all', 'projection', 'paired', 'budget'], default='all')
    args = parser.parse_args()
    if args.section in ('all', 'projection'):
        demonstrate_projection()
    if args.section in ('all', 'paired'):
        demonstrate_paired_scores()
    if args.section in ('all', 'budget'):
        demonstrate_token_budget()


if __name__ == '__main__':
    main()
