"""Reproduce a small Shannon entropy / prefix coding / prediction experiment.
Pure Python standard library. Python 3.9+.
"""
from collections import Counter, defaultdict
from heapq import heappush, heappop
from itertools import count
from math import log2
from random import Random

PROBS = {'UP': 0.5, 'DOWN': 0.25, 'LEFT': 0.125, 'RIGHT': 0.125}

def entropy(probs):
    return -sum(p * log2(p) for p in probs.values() if p > 0)

def huffman(probs):
    serial = count()
    heap = []
    for symbol, p in probs.items():
        heappush(heap, (p, next(serial), symbol))
    while len(heap) > 1:
        p1, _, left = heappop(heap)
        p2, _, right = heappop(heap)
        heappush(heap, (p1 + p2, next(serial), (left, right)))
    root = heap[0][2]
    codes = {}
    def visit(node, prefix):
        if isinstance(node, str):
            codes[node] = prefix or '0'
        else:
            visit(node[0], prefix + '0')
            visit(node[1], prefix + '1')
    visit(root, '')
    return codes

def encode(message, codes):
    return ''.join(codes[x] for x in message)

def decode(bits, codes):
    reverse = {v:k for k,v in codes.items()}
    out = []
    buffer = ''
    for bit in bits:
        buffer += bit
        if buffer in reverse:
            out.append(reverse[buffer])
            buffer = ''
    if buffer:
        raise ValueError('Truncated bitstream')
    return out

def cross_entropy(p, q):
    return -sum(p[x] * log2(q[x]) for x in p if p[x] > 0)

def text_prediction_example():
    # Tiny illustrative text sample: train bigram counts on repeated text
    text = ('the cat sat on the mat. the cat slept on the rug. '
            'the dog sat on the mat. the dog slept on the rug. ') * 40
    chars = sorted(set(text))
    counts = Counter(text)
    pairs = defaultdict(Counter)
    for prev, current in zip(text, text[1:]):
        pairs[prev][current] += 1
    n = len(chars)
    unigram_bits = -sum(log2((counts[c]+1)/(len(text)+n)) for c in text) / len(text)
    # Add-one smoothing, predicting next char from preceding char
    bigram_bits = -sum(log2((pairs[a][b]+1)/(sum(pairs[a].values())+n)) for a,b in zip(text, text[1:])) / (len(text)-1)
    return unigram_bits, bigram_bits

def main():
    rng = Random(7)
    symbols = list(PROBS)
    weights = list(PROBS.values())
    sample = rng.choices(symbols, weights=weights, k=10000)
    codes = huffman(PROBS)
    bits = encode(sample, codes)
    assert decode(bits, codes) == sample
    theoretical = entropy(PROBS)
    expected = sum(PROBS[s] * len(codes[s]) for s in PROBS)
    print('Codes:', codes)
    print(f'Fixed-length baseline: 2.00000 bits/instruction')
    print(f'Shannon entropy: {theoretical:.5f} bits/instruction')
    print(f'Expected Huffman: {expected:.5f} bits/instruction')
    print(f'Sampled Huffman: {len(bits)/len(sample):.5f} bits/instruction')
    print(f'Roundtrip decoded: {decode(bits, codes) == sample}')
    wrong = {'UP':.25, 'DOWN':.25, 'LEFT':.25, 'RIGHT':.25}
    print(f'Cross-entropy with uniform wrong model: {cross_entropy(PROBS, wrong):.5f} bits/instruction')
    uni, bi = text_prediction_example()
    print(f'Illustrative training-text code lengths: unigram={uni:.3f}, bigram={bi:.3f} bits/character')
    print('Note: training-set figures are optimistic, not generalization estimates.')

if __name__ == '__main__':
    main()
