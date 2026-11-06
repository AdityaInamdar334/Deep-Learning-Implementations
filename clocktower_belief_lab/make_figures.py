from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from beliefs import after_survival,worlds
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
data=json.loads((ROOT/'data/published_ratings.json').read_text())
ms=sorted(data['models'],key=lambda m:m['combined'])
labels={m['id']:m['id'].replace('claude-','Claude ').replace('gpt-','GPT-').replace('-astra',' Astra').replace('-sol',' Sol') for m in ms}
for a,b in [('4-6','4.6'),('4-7','4.7'),('4-8','4.8'),('5-5','5.5'),('5-1','5.1')]:labels={k:v.replace(a,b) for k,v in labels.items()}
labels={k:v.replace('opus-','Opus ').replace('fable-','Fable ') for k,v in labels.items()}
fig,ax=plt.subplots(figsize=(11,7.5),layout='constrained')
for i,m in enumerate(ms):
 c='#176b87' if m['lab']=='OpenAI' else '#a84b35'
 ax.errorbar(m['combined'],i,xerr=[[m['combined']-m['c90'][0]],[m['c90'][1]-m['combined']]],fmt='o',color=c,capsize=4,ms=7)
 ax.text(2030,i,str(m['combined']),va='center',fontsize=10)
ax.set_yticks(range(len(ms)),[labels[m['id']] for m in ms]);ax.set_xlim(1020,2090);ax.axvline(1500,color='#aab4bb',ls=':',lw=1)
ax.set_xlabel('Published combined rating on an Elo scale');ax.grid(axis='x',alpha=.15)
fig.suptitle('Clocktower: the leaderboard has uncertainty',fontsize=19,fontweight='bold')
ax.set_title('386 games • 193 side-swapped pairs • bars show published 90% bootstrap intervals',fontsize=10,pad=16)
fig.savefig(ROOT/'figures/ratings.png',dpi=190);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
vals=[after_survival(worlds(.3,a,b))*100 for a,b in [(0,0),(.1,.1),(.8,.1)]]
ax.barh(['No poisoning possible','Independent poisoning (10%)','Targeted poisoning (80% if Demon)'],vals,color=['#82909b','#176b87','#a84b35'])
ax.axvline(30,ls='--',color='#333',label='Prior: 30%');ax.set_xlim(0,35);ax.set_xlabel('Probability target is the Demon after surviving the shot (%)')
for i,v in enumerate(vals):ax.text(v+.5,i,f'{v:.1f}%',va='center')
ax.legend(loc='lower right');fig.suptitle('A failed test depends on how the test can fail',fontsize=17,fontweight='bold')
ax.set_title('Synthetic teaching example. These probabilities were not estimated from benchmark games.',fontsize=9,pad=15)
fig.savefig(ROOT/'figures/poisoning.png',dpi=190);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained');ax.set_xlim(0,11);ax.set_ylim(0,5.5);ax.axis('off')
def box(x,y,w,h,t,c):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor=c,edgecolor='none'))
 ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=12,color='#173242')
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#536b79'})
box(.4,2.1,2.8,1.1,'Observed:\ntarget survives','#e8edf1')
box(4,3.7,3,1,'Working Slayer\nTarget is not Demon','#d5e8ed')
box(4,.6,3,1,'Poisoned Slayer\nTarget may be Demon','#f4ddd5')
box(8,2.1,2.6,1.1,'Keep both explanations\nuntil distinguished','#e8edf1')
arrow((3.3,2.9),(3.9,4.1));arrow((3.3,2.4),(3.9,1.2));arrow((7.1,4.1),(7.9,2.9));arrow((7.1,1.2),(7.9,2.4))
ax.text(5.5,5.25,'One observation, multiple possible causes',ha='center',fontsize=20,fontweight='bold')
ax.text(5.5,.05,'Simplified: assumes a real Slayer, an unused ability, and a living target; other impairments omitted.',ha='center',fontsize=9)
fig.savefig(ROOT/'figures/evidence.png',dpi=190);plt.close(fig)
