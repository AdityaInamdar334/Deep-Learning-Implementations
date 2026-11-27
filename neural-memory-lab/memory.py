"""Controlled memory mechanisms, not trained model benchmarks."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
OUT=Path(__file__).parent/'figures'; OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(42)
# Write a signed bit, then process distractors; oracle gate receives write flag.
lengths=[8,32,128,512]; results={}
for T in lengths:
    b=rng.choice([-1.,1.],4096); distract=rng.normal(size=(T,4096))
    h=b.copy(); selective=b.copy()
    for x in distract:
        h=.95*h+.05*x
        selective=1*selective+0*x
    results[T]={'decay':float(np.mean(np.sign(h)==b)), 'selective':float(np.mean(np.sign(selective)==b))}
plt.figure();
for name in ['decay','selective']: plt.plot(lengths,[results[t][name] for t in lengths], 'o-',label=name)
plt.xscale('log',base=2);plt.ylim(.45,1.03);plt.xlabel('Distractor steps');plt.ylabel('Recall accuracy');plt.legend();plt.tight_layout();plt.savefig(OUT/'selective_recall.png',dpi=180);plt.close()
# Same key written repeatedly, compare additive, delta, and exact last-write store.
counts=[1,2,4,8,16,32]; overwrite={}
for n in counts:
    values=rng.choice([-1.,1.],(4096,n)); additive=values.sum(axis=1); delta=np.zeros(4096)
    for v in values.T: delta += v-delta
    overwrite[n]={'additive':float(np.mean(np.sign(additive)==values[:,-1])), 'delta':float(np.mean(delta==values[:,-1]))}
plt.figure();
for name in ['additive','delta']:plt.plot(counts,[overwrite[n][name] for n in counts],'o-',label=name)
plt.xscale('log',base=2);plt.ylim(.35,1.03);plt.xlabel('Writes to same key');plt.ylabel('Latest-value accuracy (ties incorrect)');plt.legend();plt.tight_layout();plt.savefig(OUT/'overwrite.png',dpi=180);plt.close()
# Associative prefix scan: inclusive Hillis-Steele educational implementation.
a=rng.uniform(.1,.99,(129,4)); b=rng.normal(size=(129,4)); h=np.zeros(4); serial=[]
for ai,bi in zip(a,b): h=ai*h+bi;serial.append(h.copy())
A=a.copy();B=b.copy(); offset=1
while offset<len(A):
    A0=A.copy();B0=B.copy(); A[offset:]=A0[offset:]*A0[:-offset];B[offset:]=A0[offset:]*B0[:-offset]+B0[offset:];offset*=2
err=float(np.max(np.abs(B-np.array(serial))));assert err<1e-12
out={'recall':results,'overwrite':overwrite,'scan_max_error':err};print(json.dumps(out,indent=2));(OUT.parent/'memory_results.json').write_text(json.dumps(out,indent=2))
