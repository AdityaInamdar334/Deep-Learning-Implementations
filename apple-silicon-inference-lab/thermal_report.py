#!/usr/bin/env python3
"""Extract thermal pressure and temperature lines from an Apple powermetrics log, without inventing missing sensor data."""
import argparse,json,pathlib,re
p=argparse.ArgumentParser();p.add_argument('log');p.add_argument('--output',default='results/thermal_summary.json');a=p.parse_args()
text=pathlib.Path(a.log).read_text(errors='replace');lines=[s.strip() for s in text.splitlines() if re.search(r'thermal|temperature|GPU Power|CPU Power|fan',s,re.I)]
obj={'source':a.log,'matching_lines':lines,'note':'Diagnostic raw lines only. Not a synchronized temperature/time-series measurement; availability depends on Mac hardware, OS, sampler and privileges.'}
out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,indent=2));print(f'Wrote {out} ({len(lines)} matching lines)')
