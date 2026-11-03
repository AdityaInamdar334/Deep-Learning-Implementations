"""Binary classification and operational metrics, with denominator conventions explicit."""
import math


def ratio(n, d):
    return n / d if d else None  # undefined metrics serialize as JSON null


def confusion(rows, prediction_key='security_queue', label_key='label'):
    tp = sum(bool(r[prediction_key]) and bool(r[label_key]) for r in rows)
    fp = sum(bool(r[prediction_key]) and not bool(r[label_key]) for r in rows)
    fn = sum(not bool(r[prediction_key]) and bool(r[label_key]) for r in rows)
    tn = sum(not bool(r[prediction_key]) and not bool(r[label_key]) for r in rows)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def summarize(rows):
    out = {}
    for label, stage in (('l1', 'screened'), ('l2', 'security_queue'), ('p0', 'urgent_alert')):
        c = confusion(rows, stage, 'urgent_truth' if label == 'p0' else 'label')
        out[f'{label}_recall'] = ratio(c['tp'], c['tp']+c['fn'])
        out[f'{label}_precision'] = ratio(c['tp'], c['tp']+c['fp'])
        out[f'{label}_fpr'] = ratio(c['fp'], c['fp']+c['tn'])
        out[f'{label}_tp'] = c['tp']
        out[f'{label}_fp'] = c['fp']
    out['coverage'] = sum(r['covered'] for r in rows)/len(rows)
    out['l1_fraction'] = sum(r['screened'] for r in rows)/len(rows)
    out['l2_fraction'] = sum(r['investigated'] for r in rows)/len(rows)
    out['l3_fraction'] = sum(r['l3'] for r in rows)/len(rows)
    out['mean_cost_units'] = sum(r['cost_units'] for r in rows)/len(rows)
    out['n'] = len(rows)
    out['positive_n'] = sum(r['label'] for r in rows)
    out['urgent_n'] = sum(r['urgent_truth'] for r in rows)
    return out


def mean_sd(nums):
    observed = [x for x in nums if x is not None]
    if not observed:
        return None, None
    mean = sum(observed)/len(observed)
    sd = math.sqrt(sum((n-mean)**2 for n in observed)/(len(observed)-1)) if len(observed)>1 else 0
    return mean, sd
