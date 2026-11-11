"""Synthetic softmax learning example, NOT a reimplementation of Laya/Maya."""
import math
import random
from .core import softmax


def make_rows(n=500, k=3, seed=19):
    rng=random.Random(seed)
    rows=[]
    for idx in range(n):
        features=[rng.uniform(-1,1) for _ in range(k)]
        gold=max(range(k),key=lambda j:features[j])
        # Confound train-set position with gold identity to induce a shortcut.
        perm=[gold] + [i for i in range(k) if i != gold]
        if rng.random() < .2:
            rng.shuffle(perm)
        rows.append(([features[i] for i in perm], perm.index(gold)))
    return rows


def train(rows, shuffle_options=False, epochs=12, learning_rate=.02, seed=29):
    rng=random.Random(seed)
    k=len(rows[0][0]); content_w=.15; bias=[0.0]*k
    examples=list(rows)
    for _ in range(epochs):
        rng.shuffle(examples)
        for data, gold in examples:
            data=list(data)
            if shuffle_options:
                order=list(range(k)); rng.shuffle(order)
                gold=order.index(gold)
                data=[data[j] for j in order]
            pred=softmax([content_w*x + b for x,b in zip(data,bias)])
            grad_w=sum((p-(j==gold))*x for j,(p,x) in enumerate(zip(pred,data)))
            content_w-=learning_rate*grad_w
            for j in range(k):
                bias[j]-=learning_rate*(pred[j]-(j==gold))
    return {'content_weight':content_w,'slot_bias':bias}


def predict_model(row, params):
    vals=[params['content_weight']*x + params['slot_bias'][j] for j,x in enumerate(row)]
    return max(range(len(vals)),key=lambda j:vals[j])


def eval_model(params, n=150, k=3, seed=199):
    rng=random.Random(seed)
    flips=0; accuracy=0.0
    for _ in range(n):
        vals=[rng.uniform(-1,1) for _ in range(k)]
        gold=max(range(k),key=lambda j:vals[j])
        results=[]
        for shift in range(k):
            order=list(range(k))[shift:]+list(range(k))[:shift]
            pred_slot=predict_model([vals[j] for j in order],params)
            results.append(order[pred_slot])
            accuracy += (results[-1]==gold) / (k*n)
        flips += (len(set(results))>1)
    return {'flip_rate':flips/n, 'mean_accuracy':accuracy}
