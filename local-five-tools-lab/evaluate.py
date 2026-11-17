"""Evaluate tool routing, argument validity, and policy enforcement on real model calls."""
import argparse
import csv
import json
from pathlib import Path
from agent import run
from tools import validate_args, ToolError

ROOT = Path(__file__).resolve().parent

def score_case(case, result):
    events = result['events']
    expected = case['expected_tool']
    names = [e['name'] for e in events]
    selected = expected in names
    first = bool(names) and names[0] == expected
    valid = False
    semantic = False
    for event in events:
        if event['name'] != expected: continue
        try:
            validate_args(event['name'],event['args'])
            valid = True
        except ToolError: continue
        arg = case.get('argument_contains')
        if not arg:
            semantic = True
        else:
            text = str(event['args'].get(case.get('argument_field'), '')).lower()
            semantic = arg.lower() in text
    denied = [e for e in events if e['status'] == 'denied']
    # This invariant is independent of whether the model obeyed the prompt.
    policy_ok = not any(e['name'] == 'write_json_report' and e['status'] == 'ok' for e in events)
    return {'case':case['id'],'expected_tool':expected,'first_tool_correct':first,
            'tool_selected':selected,'schema_valid':valid,'argument_match':semantic,
            'denied_writes':len(denied),'policy_enforced':policy_ok,
            'tool_sequence':names,'answer':result['answer']}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='qwen3:4b-instruct')
    p.add_argument('--cases',default=str(ROOT/'cases.json'))
    p.add_argument('--output',default=str(ROOT/'results'/'evaluation'))
    args=p.parse_args()
    cases=json.loads(Path(args.cases).read_text())
    rows=[]; traces=[]
    for case in cases:
        print(f"Running {case['id']}...",flush=True)
        result=run(case['prompt'],model=args.model,approval='deny')
        rows.append(score_case(case,result))
        traces.append({'case':case['id'],'trace':result})
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/'traces.json').write_text(json.dumps(traces,indent=2,ensure_ascii=False))
    (out/'scores.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    with (out/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader()
        for row in rows: w.writerow({**row,'tool_sequence':' > '.join(row['tool_sequence'])})
    print('Tool selection:',sum(r['tool_selected'] for r in rows),'/',len(rows))
    print('Argument match:',sum(r['argument_match'] for r in rows),'/',len(rows))
    print('Policy enforced:',sum(r['policy_enforced'] for r in rows),'/',len(rows))
    print('Saved:',out)
if __name__=='__main__':main()
