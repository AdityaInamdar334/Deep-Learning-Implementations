"""Optional: rebuild compact records from a checkout of the pinned upstream source."""
import argparse,json
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('checkout',type=Path);args=parser.parse_args()
root=args.checkout/'games';rows=[]
for pattern in ['v16/f2-*','v17/f3*-*','v17/f4-*','v17/f5-*']:
    for batch in sorted(root.glob(pattern)):
        for game in sorted(batch.glob('game-*')):
            if not (game/'result.json').exists():continue
            r=json.loads((game/'result.json').read_text());c=json.loads((game/'config.json').read_text())
            if r.get('winner') not in ('good','evil'):continue
            if 'v16' in game.parts and 'claude-opus-4-8' in (c['good_model'],c['evil_model']):continue
            inplay=set(r['starting'].values())
            rows.append({'path':str(game.relative_to(root)),'pair':batch.name+'/'+game.name.replace('-swap',''),
                         'good':c['good_model'],'evil':c['evil_model'],'won':r['winner']=='good',
                         'minions':[m in inplay for m in ['Spy','Scarlet Woman','Baron']],
                         'players':c.get('players') or len(r['starting'])})
out=Path(__file__).resolve().parent/'data/games.json'
out.write_text(json.dumps(rows,indent=2));print(f'Wrote {len(rows)} records to {out}')
