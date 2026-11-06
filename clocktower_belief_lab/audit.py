"""Re-fit the published point estimates from public game metadata; no model calls."""
import json
from pathlib import Path
from vendor.ratings import fit
ROOT=Path(__file__).resolve().parent

def run():
    games=json.loads((ROOT/'data/games.json').read_text())
    published=json.loads((ROOT/'data/published_ratings.json').read_text())
    names=[m['id'] for m in published['models']]
    combined,_,extra=fit(games,names)
    det,dec,_=fit(games,names,split=True)
    checks=[]
    for m in published['models']:
        computed=[round(combined[m['id']]),round(det[m['id']]),round(dec[m['id']])]
        expected=[m['combined'],m['detection'],m['deception']]
        checks.append(computed==expected)
    print(f'Games: {len(games)}; evil wins: {sum(not g["won"] for g in games)}')
    print(f'All published rounded rating point estimates match: {all(checks)}')
    print('Published intervals are provided, not re-estimated by this command.')
    print('Two model-level summaries from public metadata:')
    for name in ['gpt-6-astra','gpt-6.1-sol']:
        good=[g for g in games if g['good']==name]; evil=[g for g in games if g['evil']==name]
        print(f'  {name}: good {sum(g["won"] for g in good)}/{len(good)}; evil {sum(not g["won"] for g in evil)}/{len(evil)}')
    return games,published,all(checks)
if __name__=='__main__':
    _,_,ok=run()
    if not ok:raise SystemExit('Rating mismatch')
