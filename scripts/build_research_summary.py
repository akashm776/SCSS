"""Render the public D1 figure and tables from archived comparison rows (no training)."""
import json
import os
from pathlib import Path
import statistics
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'otco-paper-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def build():
    d=json.loads((ROOT/'experiment_results/clip_source_complete_2026-10-16/comparison.json').read_text())
    effect=d['timecourse']['50']['effects']
    arms=[f'{r}_{a}' for r in ['H','M','C','I'] for a in ['Select','Synth']]
    for w in ['w100','w500']:
        for a in arms:
            for metric in ['loss','r1_pp']:
                mean=statistics.fmean(s['timecourse']['50']['effects'][w][a][metric] for s in d['per_seed'])
                assert abs(mean-effect[w][a][metric])<1e-14
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'scss-d1'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.7),sharey=True)
    colors=['#17789c','#bd5427'];y=np.arange(len(arms))
    for ax,metric,factor,title,label in zip(axes,['loss','r1_pp'],[1000,1],['Reporting loss','Bidirectional retrieval (R@1)'],['Treatment − native loss (×10⁻³)\n← lower is better','Treatment − native R@1 (percentage points)\nhigher is better →']):
        ax.axvline(0,color='#555555',linewidth=1)
        for w,offset,color,legend in zip(['w100','w500'],[-.15,.15],colors,['Early weights (100)','Late weights (500)']):
            values=[effect[w][a][metric]*factor for a in arms]
            for k,s in enumerate(d['per_seed']):
                v=[s['timecourse']['50']['effects'][w][a][metric]*factor for a in arms]
                ax.scatter(v,y+offset+(k-1)*.035,s=16,facecolors='none',edgecolors=color,alpha=.5,linewidths=.8)
            ax.scatter(values,y+offset,s=46,c=color,marker='D',label=legend,zorder=3)
        ax.set_title(title,fontweight='bold',loc='left',pad=12)
        ax.set_xlabel(label);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.set_yticks(y, [a.replace('_','-').replace('Synth','Synthesis') for a in arms]);ax.tick_params(axis='y',length=0)
    axes[0].invert_yaxis();axes[1].legend(loc='lower right',fontsize=9,frameon=True)
    fig.suptitle('D1: early loss gains do not imply retrieval gains',fontsize=15,fontweight='bold')
    fig.subplots_adjust(left=.11, right=.985, bottom=.20, top=.84, wspace=.10)
    fig.text(.02,.025,'Horizon 50 · Diamonds: equal-seed means · Open circles: three seed means (two streams averaged per seed)\nNo confidence intervals are implied. Source regimes and controls are defined in the report.',fontsize=9,color='#444444')
    dest=ROOT/'docs/figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'d1_research_summary.svg',metadata={'Date':None})
    fig.savefig(dest/'d1_research_summary.png',dpi=160);plt.close(fig)
    lines=['# Appendix A. Detailed D1 results','', '[Report contents](../../README.md#contents) · [Results](results.md) · [Evidence index](evidence.md)','',
           'These tables are generated from the [completed D1 comparison](../../experiment_results/clip_source_complete_2026-10-16/comparison.json). Streams are averaged within each seed; the three seeds receive equal weight. H50 is the frozen primary horizon. Loss differences are unscaled; retrieval differences are percentage points.', '',
           '| Arm | Early loss Δ | Late loss Δ | Early R@1 Δ (pp) | Late R@1 Δ (pp) |','|---|---:|---:|---:|---:|']
    for a in arms:
        x,z=effect['w100'][a],effect['w500'][a]
        lines.append(f"| {a.replace('_','-')} | {x['loss']:+.9f} | {z['loss']:+.9f} | {x['r1_pp']:+.5f} | {z['r1_pp']:+.5f} |")
    lines += ['','## Prespecified side interaction','',
              'Q = D(C-Select) − D(I-Select). The interaction is Q(late) − Q(early); negative favors the proposed relative shift toward C.', '',
              '| Seed | Q early | Q late | Interaction |','|---|---:|---:|---:|']
    for seed in d['per_seed']:
        c=seed['timecourse']['50']['contrasts']
        lines.append(f"| {seed['training_seed']} | {c['Q']['w100']['loss']:+.9f} | {c['Q']['w500']['loss']:+.9f} | {c['Delta_side']['loss']:+.9f} |")
    c=d['timecourse']['50']['contrasts']
    lines.append(f"| Equal-seed mean | {c['Q']['w100']['loss']:+.9f} | {c['Q']['w500']['loss']:+.9f} | {c['Delta_side']['loss']:+.9f} |")
    lines += ['', '## Secondary synthesis contrast','', 'Synth − Select within each source regime; negative favors synthesis.', '',
              '| Regime | Early loss contrast | Late loss contrast |','|---|---:|---:|']
    for r in ['H','M','C','I']:
        lines.append(f"| {r} | {c['G'][r]['w100']['loss']:+.9f} | {c['G'][r]['w500']['loss']:+.9f} |")
    lines += ['', 'The full comparison retains stream-level matches and all recorded horizons. These summaries do not imply statistical significance. See the [limitations](limitations.md) and [completed interpretation](../research-status-2026-10-16-source-complete.md).', '',
              'Regenerate this appendix and the figure with `python -m scripts.build_research_summary`. Matplotlib is required for rendering.','']
    (ROOT/'docs/paper/detailed-results.md').write_text('\n'.join(lines))
    print('Verified equal-seed arithmetic; wrote D1 figure and detailed results.')
if __name__=='__main__':build()
