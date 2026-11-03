"""Model-free deterministic policy simulators and a strict tool registry.

No real LLM is called. The 'improviser' is a deliberately flawed scripted policy.
"""
from dataclasses import dataclass, field
from typing import Any
from hashlib import sha256
import json
import numpy as np
from .data import Recording
from .methods import circular_shift_surrogates, population_coupling, pca_variance, bh_adjust, bin_spikes, sttc

ALLOWED_TOOLS = {'slice_conditions', 'pairwise_sttc', 'coupling', 'pca', 'provenance'}

class ClarificationRequired(ValueError):
    pass

@dataclass
class ToolSession:
    recording: Recording
    calls: list = field(default_factory=list)

    def call(self, tool: str, **kwargs):
        if tool not in ALLOWED_TOOLS:
            raise PermissionError(f'Tool not allowlisted: {tool}')
        if tool=='slice_conditions':
            windows=kwargs.get('windows')
            if not isinstance(windows,dict) or set(windows)!={'cue','no_cue'}:
                raise ClarificationRequired('Define cue and no_cue recording intervals explicitly.')
            if any(not isinstance(v,list) or not v for v in windows.values()):
                raise ClarificationRequired('Each condition needs nonempty intervals')
            for intervals in windows.values():
                for lo,hi in intervals:
                    if not (0<=lo<hi<=self.recording.duration_ms): raise ValueError('invalid epoch')
            self.calls.append({'tool':tool,'windows':windows})
            return windows
        if tool=='pairwise_sttc':
            sub=kwargs['recording']
            delta=float(kwargs.get('delta_ms',20))
            ns=int(kwargs.get('surrogates',39))
            seed=int(kwargs.get('seed',0))
            if not (0 <= ns <= 999) or not (0 < delta <= 100):
                raise ValueError('bounded, validated parameters required')
            results=[]
            for i in range(sub.n_units):
                for j in range(i+1,sub.n_units):
                    r=(circular_shift_surrogates(sub.spikes[i],sub.spikes[j],sub.duration_ms,
                                                delta_ms=delta,n_surrogates=ns,seed=seed + 1009*i + j)
                       if ns else {'observed':float(sttc(sub.spikes[i],sub.spikes[j],sub.duration_ms,delta)),
                                   'p_two_sided':None,'null_mean':None})
                    results.append({'pair':[i,j],**r})
            q=bh_adjust([float(r['p_two_sided']) if r['p_two_sided'] is not None else np.nan for r in results])
            for r,v in zip(results,q):r['q_bh']=float(v) if np.isfinite(v) else None
            self.calls.append({'tool':tool,'delta_ms':delta,'surrogates':ns,'pairs':len(results)})
            return results
        if tool=='coupling':
            sub=kwargs['recording']
            bin_ms=float(kwargs.get('bin_ms',50))
            if not (10<=bin_ms<=250):raise ValueError('invalid bin width')
            ans=population_coupling(sub,bin_ms=bin_ms)
            self.calls.append({'tool':tool,'bin_ms':bin_ms})
            return [float(x) if np.isfinite(x) else None for x in ans]
        if tool=='pca':
            sub=kwargs['recording']
            bin_ms=float(kwargs.get('bin_ms',50))
            if not (10<=bin_ms<=250):raise ValueError('invalid bin width')
            ans=pca_variance(sub,bin_ms=bin_ms)
            self.calls.append({'tool':tool,'bin_ms':bin_ms})
            return ans
        if tool=='provenance':
            self.calls.append({'tool':tool})
            return {'input_sha256':self.recording.digest(),'version':'0.1.0','time_unit':'ms'}


def _aggregate_conditions(recording, windows):
    """Concatenate intervals with time reset; never concatenate across a gap for STTC.

    To avoid artificial adjacencies, choose a single contiguous specified window
    for STTC in this small lab; multi-interval workflows need per-epoch STTC.
    """
    if any(len(v)!=1 for v in windows.values()):
        raise ValueError('educational runner accepts single contiguous interval per condition')
    return {k:recording.window(*v[0]) for k,v in windows.items()}


def correct_windows(recording):
    a,b=recording.metadata['cue_ms']
    # One contiguous no-cue interval, not the entire complement; explicitly stated.
    return {'cue':[[a,b]], 'no_cue':[[0,a]]}


def run_policy(recording: Recording, policy='bounded', seed=0, surrogates=19):
    """Run specified scripted behavior. 'improvised' contains intentional failures."""
    if policy not in {'bounded','improvised','no_verification','silent_downsample'}:
        raise ValueError(policy)
    before=recording.digest()
    session=ToolSession(recording)
    clarified=False
    if policy in {'bounded','no_verification','silent_downsample'}:
        try:
            session.call('slice_conditions', windows=None)
        except ClarificationRequired:
            clarified=True
        windows=session.call('slice_conditions',windows=correct_windows(recording))
    else:
        # Unreported inference: first half vs second half, not actual cue/no-cue.
        mid=recording.duration_ms/2
        windows={'cue':[[0,mid]],'no_cue':[[mid,recording.duration_ms]]}
    parts=_aggregate_conditions(recording,windows)
    summary={}
    for condition, sub in parts.items():
        if policy in {'bounded','no_verification','silent_downsample'}:
            pair_results=session.call('pairwise_sttc',recording=sub,delta_ms=20,
                                      surrogates=(0 if policy=='no_verification' else surrogates),seed=seed)
            effective_bin=200 if policy=='silent_downsample' else 50
            coupling=session.call('coupling', recording=sub,bin_ms=effective_bin)
            pca=session.call('pca',recording=sub,bin_ms=effective_bin)
            measure='sttc'
        else:
            # Deliberate methodological improvisation: replace STTC with
            # correlation of binned rates; silently use coarser temporal bins,
            # and include own activity in population coupling.
            effective_bin=150
            counts=bin_spikes(sub,effective_bin)
            corr=np.corrcoef(counts)
            pair_results=[{'pair':[i,j],'observed':float(corr[i,j]) if np.isfinite(corr[i,j]) else None,
                           'p_two_sided':None,'q_bh':None}
                          for i in range(sub.n_units) for j in range(i+1,sub.n_units)]
            total=counts.sum(axis=0)
            coupling=[float(np.corrcoef(counts[i],total)[0,1]) if np.std(counts[i])>0 else None
                      for i in range(sub.n_units)]
            pca=pca_variance(sub,effective_bin)
            measure='binned_pearson_mislabeled_as_sttc'
        summary[condition]={
            'duration_ms':sub.duration_ms,
            'unit_count':sub.n_units,
            'spike_count':sub.total_spikes,
            'pair_results':pair_results,
            'population_coupling':coupling,
            'pca_explained_variance':pca,
            'effective_bin_ms':effective_bin,
            'pair_measure':measure,
        }
    provenance=(session.call('provenance') if policy in {'bounded','no_verification','silent_downsample'} else None)
    payload={
        'policy':policy,
        'conditions':windows,
        'clarification_requested':clarified,
        'source_unchanged':before==recording.digest(),
        'analysis':summary,
        'provenance':provenance,
        'audit_log':session.calls,
        'caveats':(['Synthetic data; no biological causal inference.','STTC reflects spike-time association, not synaptic connectivity.','PCA is not a latent state-space model.']
                   if policy in {'bounded','no_verification','silent_downsample'} else []),
    }
    return payload
