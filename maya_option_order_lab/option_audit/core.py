"""Order-sensitivity statistics. Predictions must reference stable OPTION IDs, not slots."""
import math
import random
from collections import defaultdict


def cyclic_rotations(ids):
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('option IDs must be nonempty and unique')
    return [ids[i:] + ids[:i] for i in range(len(ids))]


def sampled_orderings(ids, count=6, seed=1):
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)) or count < 1:
        raise ValueError('invalid ids or count')
    rng = random.Random(seed)
    seen = set()
    out = []
    maximum = math.factorial(len(ids))
    target = min(count, maximum)
    while len(out) < target:
        cand = list(ids)
        rng.shuffle(cand)
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def softmax(xs):
    if not xs:
        raise ValueError('scores cannot be empty')
    top = max(xs)
    z = [math.exp(x - top) for x in xs]
    total = sum(z)
    return [x / total for x in z]


def predict(scores, order):
    if set(scores) != set(order) or len(scores) != len(order):
        raise ValueError('score keys must equal option IDs')
    # deterministic ID-stable tie break: avoid hiding order bias in argmax behavior
    return sorted(scores, key=lambda k: (-scores[k], k))[0]


def total_variation(probs):
    k = len(probs)
    if k == 0:
        raise ValueError('empty distribution')
    if any(x < 0 for x in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-7):
        raise ValueError('invalid probability distribution')
    return 0.5 * sum(abs(p - 1.0 / k) for p in probs)


def audit_records(rows, bootstrap=500, seed=9):
    """rows: list of {id, gold, presentations:[{order:[IDs], scores:{ID:float}}]}."""
    if not rows:
        raise ValueError('empty audit')
    checks = []
    slot_correct, slot_count = defaultdict(int), defaultdict(int)
    for item in rows:
        presentations = item['presentations']
        if len(presentations) < 2:
            raise ValueError('need at least two orderings per item')
        canonical = set(presentations[0]['order'])
        if item['gold'] not in canonical:
            raise ValueError('gold label must belong to options')
        labels, correct, vote = [], [], defaultdict(float)
        for p in presentations:
            order, scores = p['order'], p['scores']
            if len(order) != len(canonical) or set(order) != canonical:
                raise ValueError('each ordering must be a permutation')
            best = predict(scores, order)
            labels.append(best)
            correct.append(best == item['gold'])
            slot = order.index(item['gold'])
            slot_count[slot] += 1
            slot_correct[slot] += (best == item['gold'])
            # confidence-weighted plurality: vote for each predicted ID weighted by p(top)
            for identity, value in scores.items():
                if not math.isfinite(value):
                    raise ValueError('non-finite scores')
            probs = softmax([scores[k] for k in order])
            vote[best] += max(probs)
        vote_best = sorted(vote, key=lambda k: (-vote[k], k))[0]
        checks.append({'id': item['id'], 'flips': len(set(labels)) > 1,
                       'accuracy': sum(correct) / len(correct),
                       'vote_correct': vote_best == item['gold'],
                       'predictions': labels})
    def aggregate(sample):
        return sum(item['flips'] for item in sample) / len(sample)
    rng = random.Random(seed)
    observed = aggregate(checks)
    bounds = []
    for _ in range(bootstrap):
        bounds.append(aggregate([rng.choice(checks) for _ in checks]))
    bounds.sort()
    if bounds:
        ci = [bounds[int(.025 * (len(bounds)-1))], bounds[int(.975 * (len(bounds)-1))]]
    else:
        ci = [observed, observed]
    return {'items': len(checks), 'flip_rate': observed, 'bootstrap_ci95': ci,
            'mean_accuracy': sum(x['accuracy'] for x in checks) / len(checks),
            'vote_accuracy': sum(x['vote_correct'] for x in checks) / len(checks),
            'slot_accuracy': {str(s): slot_correct[s] / slot_count[s] for s in sorted(slot_count)},
            'slot_counts': {str(s): slot_count[s] for s in sorted(slot_count)},
            'unstable_items': [x['id'] for x in checks if x['flips']], 'details': checks}


def make_toy_questions(n=240, seed=4):
    rng = random.Random(seed)
    out=[]
    for index in range(n):
        k = [3, 5, 12][index % 3]
        identities=[f'option_{j}' for j in range(k)]
        # Each question has a unique, content-based best answer with intentionally close runners-up.
        vals=[rng.uniform(-0.7, 0.7) for _ in range(k)]
        gold=rng.randrange(k)
        vals[gold]=max(vals)+rng.uniform(0.03, 0.65)
        ordered=list(zip(identities, vals))
        rng.shuffle(ordered)
        options=[{'id':i, 'signal':v, 'text':f'Synthetic description {i}'} for i,v in ordered]
        out.append({'id':f'toy_{index:04d}', 'gold':identities[gold], 'options':options})
    return out


def synthetic_scores(options, order, strength=0.0):
    """Demonstration only: content score + a slot-dependent additive logit."""
    signal={o['id']:o['signal'] for o in options}
    k=len(order)
    # position 0 is advantaged; some mid slots penalized; last receives a weak bonus
    prior=[1.0 if idx == 0 else (0.28 if idx == k-1 else -0.25) for idx in range(k)]
    return {key: signal[key]+strength*prior[index] for index,key in enumerate(order)}


def toy_audit(questions, strength, seed=1):
    out=[]
    for q in questions:
        ids=[o['id'] for o in q['options']]
        orders=cyclic_rotations(ids) if len(ids)<=5 else sampled_orderings(ids, 6, seed=seed)
        out.append({'id':q['id'], 'gold':q['gold'], 'presentations':[
            {'order':p, 'scores':synthetic_scores(q['options'],p,strength)} for p in orders]})
    return audit_records(out)


def identical_options_slot_prior(k, strength, rounds=100, seed=21):
    ids=[f'option_{i}' for i in range(k)]
    rng=random.Random(seed)
    slots=[0.0]*k
    for _ in range(rounds):
        perm=list(ids)
        rng.shuffle(perm)
        scores=synthetic_scores([{'id':v, 'signal':0} for v in ids],perm,strength)
        probs=softmax([scores[k] for k in perm])
        for i,p in enumerate(probs):
            slots[i]+=p/rounds
    return {'probabilities':slots, 'tv_distance':total_variation(slots),
            'max_min_ratio':max(slots)/min(slots)}
