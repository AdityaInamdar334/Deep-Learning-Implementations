from .data import make_study
from .orchestrator import execute

def main():
    st=make_study(seed=0)
    for mode, w in [('generalist',1), ('distributed',1),('distributed',6)]:
        report=execute(st,mode=mode,workers=w,latency_s=0.02)
        print(f"{mode:11} workers={w} examined={report['features_examined']} "
              f"accepted={report['accepted_claims']} replicated={report['independently_replicated_claims']} "
              f"known-null={report['known_null_claims']} time={report['runtime_seconds']:.3f}s")
        print('Representative accepted findings:')
        for f in [x for x in report['claims'] if x['accepted']][:3]:
            print(f"  {f['name']}: discovery p={f['discovery_p']:.2g}; "
                  f"holdout p={f['holdout_p']:.2g}; d={f['effect_d']:.2f}; "
                  f"provenance={f['provenance']}")

if __name__ == '__main__':
    main()
