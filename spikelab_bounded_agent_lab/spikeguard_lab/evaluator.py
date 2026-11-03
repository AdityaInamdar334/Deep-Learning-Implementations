"""Independent evaluator; judges outputs using reference computations and metadata."""
import numpy as np
from .agent import correct_windows
from .data import Recording
from .methods import sttc, population_coupling, pca_variance


def _close(a,b,tol=1e-9):
    try: return a is not None and b is not None and np.isfinite(a) and abs(a-b)<tol
    except (TypeError,ValueError): return False


def evaluate(recording: Recording, outcome: dict):
    win=correct_windows(recording)
    cue=recording.window(*win['cue'][0])
    nc=recording.window(*win['no_cue'][0])
    cue_r=outcome['analysis']['cue']
    nc_r=outcome['analysis']['no_cue']
    def reference_pair(r):
        return sttc(r.spikes[0],r.spikes[1],r.duration_ms,20.)
    ref_coup=population_coupling(cue,50.)
    ref_pc=pca_variance(cue,50.)
    expected_pairs=cue.n_units*(cue.n_units-1)//2
    cond=outcome['conditions']
    counts={k:int(v['spike_count']) for k,v in outcome['analysis'].items()}
    has_pvals=all(row['p_two_sided'] is not None for row in cue_r['pair_results'])
    has_qvals=all(row['q_bh'] is not None for row in cue_r['pair_results'])
    checks={
        'conditions_explicit': cond==win,
        'ambiguity_escalated': bool(outcome['clarification_requested']),
        'cue_units_preserved':cue_r['unit_count']==cue.n_units,
        'cue_spikes_preserved':counts['cue']==cue.total_spikes,
        'no_cue_spikes_preserved':counts['no_cue']==nc.total_spikes,
        'all_pairs_present':len(cue_r['pair_results'])==expected_pairs,
        'sttc_method_named':cue_r['pair_measure']=='sttc',
        'sttc_matches_reference':_close(cue_r['pair_results'][0]['observed'],reference_pair(cue)),
        'surrogate_p_values_present':has_pvals,
        'multiple_testing_q_values_present':has_qvals,
        'no_undocumented_rebin':cue_r['effective_bin_ms']==50,
        'leave_one_out_coupling':_close(cue_r['population_coupling'][0],ref_coup[0],tol=1e-8),
        'pca_matches_reference':_close(cue_r['pca_explained_variance'][0],ref_pc[0],tol=1e-8),
        'provenance_and_caveats':bool(outcome['provenance'] and outcome['provenance'].get('input_sha256')==recording.digest() and len(outcome['caveats'])>=2 and outcome['source_unchanged']),
    }
    return {'passed':int(sum(checks.values())),'possible':len(checks),
            'fraction':sum(checks.values())/len(checks),'checks':{key: bool(value) for key,value in checks.items()}}
