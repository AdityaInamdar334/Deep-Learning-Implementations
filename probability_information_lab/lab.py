"""Probability & information theory companion lab. Deterministic synthetic examples."""
from pathlib import Path
import argparse, json
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def binary_entropy(p):
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)): raise ValueError('p must be in [0,1]')
    a = np.zeros_like(p)
    mask = (p > 0) & (p < 1)
    a[mask] = -(p[mask]*np.log2(p[mask]) + (1-p[mask])*np.log2(1-p[mask]))
    return a


def kl(p,q):
    p,q = np.asarray(p,float),np.asarray(q,float)
    if p.shape != q.shape or np.any(p<0) or np.any(q<0) or not np.isclose(p.sum(),1) or not np.isclose(q.sum(),1):
        raise ValueError('invalid probability distributions')
    if np.any((p>0) & (q==0)): return np.inf
    mask=p>0
    return float(np.sum(p[mask]*np.log(p[mask]/q[mask])))


def posterior_positive(prevalence=.01, sensitivity=.95, specificity=.90):
    numerator=prevalence*sensitivity
    return numerator/(numerator+(1-prevalence)*(1-specificity))


def nll(logits,labels,temperature=1):
    if temperature <= 0: raise ValueError('temperature must be positive')
    z=np.asarray(logits,float)/temperature
    y=np.asarray(labels,float)
    return float(np.mean(np.logaddexp(0,z)-y*z))


def brier(probs,labels):
    return float(np.mean((np.asarray(probs)-np.asarray(labels))**2))


def ece(probs,labels,n_bins=10):
    p=np.asarray(probs,float); y=np.asarray(labels,int)
    edges=np.linspace(0,1,n_bins+1)
    error=0.0
    for j in range(n_bins):
        m=(p>=edges[j]) & ((p<edges[j+1]) if j<n_bins-1 else (p<=edges[j+1]))
        if m.any(): error+=float(m.mean())*abs(float(y[m].mean()-p[m].mean()))
    return error


def reliability(probs,labels,n_bins=10):
    p=np.asarray(probs,float); y=np.asarray(labels,int)
    xs=[];ys=[];ns=[]
    edges=np.linspace(0,1,n_bins+1)
    for j in range(n_bins):
        m=(p>=edges[j]) & ((p<edges[j+1]) if j<n_bins-1 else (p<=edges[j+1]))
        if m.any():xs.append(float(p[m].mean()));ys.append(float(y[m].mean()));ns.append(int(m.sum()))
    return xs,ys,ns


def make_calibration_data(seed=12, n=12000):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=n)
    true_logits=1.4*x-0.3
    labels=rng.binomial(1,sigmoid(true_logits))
    # Simulated confident classifier: same decision boundary, stretched logits.
    overconfident_logits=2.6*true_logits
    return overconfident_logits,labels


def fit_temperature(logits,labels):
    candidates=np.exp(np.linspace(np.log(.25),np.log(8),600))
    losses=np.array([nll(logits,labels,t) for t in candidates])
    return float(candidates[np.argmin(losses)])


def run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Bayes base rates
    prevalences=np.logspace(-3,-.3,200)
    post=np.array([posterior_positive(float(p)) for p in prevalences])
    fig,ax=plt.subplots(figsize=(8.3,4.8));ax.plot(prevalences*100,post*100,lw=2);ax.axvline(1,ls='--',alpha=.65)
    ax.scatter([1],[posterior_positive()*100],zorder=3)
    ax.set_xscale('log');ax.set_xlabel('Prevalence (%)');ax.set_ylabel('P(condition | positive test) (%)')
    ax.set_title('A 95%-sensitive test does not imply 95% posterior probability')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'bayes_base_rates.png',dpi=170);plt.close(fig)
    # entropy and KL
    ps=np.linspace(0,1,501); h=binary_entropy(ps)
    p=np.array([.48,.04,.48]);q=np.array([.8,.1,.1])
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(ps,h,lw=2);axes[0].set(xlabel='Bernoulli probability p',ylabel='Entropy (bits)',title='Uncertainty is greatest at p=0.5');axes[0].grid(alpha=.2)
    xx=np.arange(3);axes[1].bar(xx-.18,p,.36,label='P');axes[1].bar(xx+.18,q,.36,label='Q');axes[1].set_xticks(xx,['A','B','C']);axes[1].set(title=f'KL(P||Q)={kl(p,q):.3f}, KL(Q||P)={kl(q,p):.3f} nats',ylabel='Probability');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'entropy_and_kl.png',dpi=170);plt.close(fig)
    # calibration
    logits,y=make_calibration_data();val_logits,test_logits=logits[:6000],logits[6000:];val_y,test_y=y[:6000],y[6000:]
    temp=fit_temperature(val_logits,val_y)
    raw=sigmoid(test_logits);cal=sigmoid(test_logits/temp)
    fig,ax=plt.subplots(figsize=(7,5.6));ax.plot([0,1],[0,1],ls='--',color='gray',label='Perfect calibration')
    for pp,lab in [(raw,'Overconfident'),(cal,f'Temperature scaled (T={temp:.2f})')]:
        cx,cy,_=reliability(pp,test_y)
        ax.plot(cx,cy,marker='o',lw=2,label=lab)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Calibration on held-out synthetic data');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'calibration.png',dpi=170);plt.close(fig)
    # Change variables density
    rng=np.random.default_rng(42);x=rng.uniform(0,1,75000);y2=x*x
    u=np.linspace(.015,1,500)
    fig,ax=plt.subplots(figsize=(8,4.7));ax.hist(y2,bins=75,density=True,alpha=.45,label='Observed Y=X²')
    ax.plot(u,1/(2*np.sqrt(u)),lw=2,label='Change-of-variables density')
    ax.set(xlabel='y',ylabel='Probability density',title='Uniform samples become non-uniform after squaring');ax.legend();fig.tight_layout();fig.savefig(out/'density_change.png',dpi=170);plt.close(fig)
    metrics={
        'example_test':{'prevalence':.01,'sensitivity':.95,'specificity':.9,'posterior_positive':posterior_positive()},
        'information':{'bernoulli_entropy_half_bits':float(binary_entropy(np.array([.5]))[0]),'kl_p_q_nats':kl(p,q),'kl_q_p_nats':kl(q,p)},
        'calibration':{'temperature_fitted_on_validation':temp,'test_samples':int(len(test_y)),'uncalibrated':{'log_loss':nll(test_logits,test_y),'brier':brier(raw,test_y),'ece_10_bins':ece(raw,test_y),'accuracy':float(np.mean((raw>=.5)==test_y))},'calibrated':{'log_loss':nll(test_logits,test_y,temp),'brier':brier(cal,test_y),'ece_10_bins':ece(cal,test_y),'accuracy':float(np.mean((cal>=.5)==test_y))}},
        'description':'All results are synthetic educational examples, not medical or deployed-model validation.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='figures');args=ap.parse_args();run(args.out)
