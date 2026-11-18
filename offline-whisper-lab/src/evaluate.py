#!/usr/bin/env python3
"""Auditable word error rate, not character error or hallucination grading."""
import argparse,json
from pathlib import Path
from common import word_errors

def main():
    p=argparse.ArgumentParser(description='Compute basic normalized WER from local transcript and reference.')
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--hypothesis',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    metrics=word_errors(a.reference.read_text(encoding='utf-8'),a.hypothesis.read_text(encoding='utf-8'))
    print(json.dumps(metrics,indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
