"""Replay exact, session-private growing prefixes. No external packages."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
import random
import statistics


@dataclass(frozen=True)
class Turn:
    time: float
    session: int
    intent: str
    number: int
    previous: int
    added: int
    reply: int


def generate(seed, sessions, swapped=False):
    rng = random.Random(seed)
    events = []
    start = 0.0
    for session in range(sessions):
        start += rng.expovariate(1.0)
        now = start
        intent = rng.choice(["debug", "lookup"])
        iterative = (intent == "debug") != swapped
        previous = 0
        for number in range(1, 7):
            added = rng.randint(16, 80)
            reply = rng.randint(16, 96)
            events.append(Turn(now, session, intent, number, previous, added, reply))
            previous += added + reply
            if rng.random() >= (.82 if iterative else .15):
                break
            now += rng.lognormvariate(math.log(10 if iterative else 3), .55)
    return sorted(events, key=lambda e: e.time)


def fit(history, grouped=True):
    counts = defaultdict(int)
    delays = defaultdict(list)
    last = {}
    for e in history:
        label = e.intent if grouped else "*"
        counts[label, e.number] += 1
        if e.session in last:
            prev = last[e.session]
            delays[label, prev.number].append(e.time - prev.time)
        last[e.session] = e
    stats = {}
    for state, count in counts.items():
        samples = delays[state]
        window = math.exp(statistics.mean(map(math.log, samples))) if samples else 0
        stats[state] = (window, len(samples) / count)
    return stats


def priority(policy, entry, now, stats):
    born, label, number, size = entry
    if policy == "lru":
        return (born,)
    key = (label if policy == "intent" else "*", number)
    window, chance = stats.get(key, (0, 0))
    return (now - born < window, chance, size)


def replay(events, capacity, policy, stats):
    cache = {}
    occupied = hits = inputs = 0
    fractions = []
    for e in events:
        old = cache.pop(e.session, None)
        hit = old[3] if old else 0
        assert hit <= e.previous
        occupied -= hit
        length = e.previous + e.added
        hits += hit
        inputs += length
        fractions.append(hit / length)
        size = length + e.reply
        cache[e.session] = (e.time, e.intent, e.number, size)
        occupied += size
        while occupied > capacity:
            victim = min(cache, key=lambda sid: (priority(policy, cache[sid], e.time, stats), sid))
            occupied -= cache.pop(victim)[3]
    return hits, inputs, statistics.median(fractions)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capacity", type=int, default=2000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--swap-behavior", action="store_true")
    args = p.parse_args()
    if args.capacity < 0:
        p.error("capacity must be nonnegative")
    history = generate(args.seed, 1000)
    trace = generate(args.seed + 1, 1200, args.swap_behavior)
    print(f"{len(trace)} requests, {args.capacity} cached tokens")
    for policy in ("lru", "turn", "intent"):
        stats = fit(history, grouped=policy == "intent")
        hits, total, median = replay(trace, args.capacity, policy, stats)
        print(f"{policy:8} {hits:8}/{total} tokens   pooled={100*hits/total:.2f}%   median={100*median:.2f}%")
