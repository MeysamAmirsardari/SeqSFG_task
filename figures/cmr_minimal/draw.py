from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Arial','font.size':13,'axes.titlesize':16,'axes.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','text.color':'#222222','axes.labelcolor':'#222222'})
GRAY='#b9b9b9'; INK='#171717'
figs=[]
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    figs.append(fig)
def raster(ax, offsets, final_sync=True, target=True, repeats=5):
    for j in range(repeats):
        for b,o in enumerate(offsets):
            t=50+250*j+(0 if final_sync and j==repeats-1 else o)
            ax.add_patch(Rectangle((t,b-.21),187.5,.42,facecolor=GRAY,edgecolor='none'))
    last=50+250*(repeats-1)
    if target: ax.plot([last,last+187.5],[2,2],color=INK,lw=3,solid_capstyle='butt')
    ax.set(xlim=(0,1300),ylim=(-.6,4.75),yticks=range(5),yticklabels=['250','500','1000','2000','4000'],xticks=[0,500,1000],xlabel='Time (ms)')
    ax.tick_params(length=3); ax.spines[['top','right']].set_visible(False)

# 1: exact event timing in current implementation, categorical frequency rows.
fig,axs=plt.subplots(3,1,figsize=(10,8.2),sharex=True)
for ax,title,off in zip(axs,['Synchronous','Ascending · 20 ms step','Zigzag · 20 ms step'],[[0]*5,[-40,-20,0,20,40],[40,-20,0,20,-40]]):
    raster(ax,off); ax.set_title(title,loc='left',pad=9); ax.set_xlabel('')
    ax.axvline(1010,color='#777777',lw=.8,ls=':')
axs[1].set_ylabel('Band centre (Hz; rows equally spaced)')
axs[-1].set_xlabel('Time (ms)')
fig.text(.12,.975,'Current implementation',fontsize=19)
fig.text(.12,.935,'Grey: noise bands     Black: target tone     Final burst: synchronous',fontsize=12)
fig.subplots_adjust(left=.12,right=.98,top=.88,bottom=.08,hspace=.55)
save(fig,'01_sequences')

# 2: schematic envelopes (not measured or noise waveforms).
fig,axs=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
x=np.linspace(0,1,700)
rng=np.random.default_rng(14)
def shape():
    phase=rng.uniform(0,2*np.pi,4)
    a=.48+sum(.09*np.sin(2*np.pi*(i+1)*x+p) for i,p in enumerate(phase))
    return a*np.sin(np.pi*x)**.35
shared=shape()
for ax,title,co in zip(axs,['Comodulated','Independent'],[True,False]):
    for b in range(5):
        a=shared if co else shape()
        ax.plot(x,b+a*.8,color=INK,lw=1.3)
    ax.set(xlim=(0,1),ylim=(-.05,4.8),yticks=range(5),yticklabels=['250','500','1000','2000','4000'],xticks=[],xlabel='Time within burst')
    ax.set_title(title,loc='left'); ax.spines[['top','right','bottom']].set_visible(False)
axs[0].set_ylabel('Band centre (Hz)')
fig.text(.5,.03,'CMR = threshold (independent) − threshold (comodulated)',ha='center',fontsize=14)
fig.text(.98,.96,'Envelope schematic',ha='right',fontsize=11,color='#666666')
fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.82,wspace=.2)
save(fig,'02_envelopes')

# 3: task with one signal interval, no invented performance data.
fig,axs=plt.subplots(1,3,figsize=(11,3.2),sharey=True)
for i,ax in enumerate(axs):
    raster(ax,[-40,-20,0,20,40],target=(i==1))
    ax.set_title(f'Interval {i+1}',loc='left'); ax.set_xticks([]); ax.set_xlabel(''); ax.set_yticks([])
    ax.spines[['left','bottom']].set_visible(False)
fig.text(.5,.91,'Which interval contained the tone?',ha='center',fontsize=19)
fig.text(.5,.05,'Example answer: 2     ·     Vary tone level to estimate threshold',ha='center',fontsize=13)
fig.subplots_adjust(left=.03,right=.98,top=.73,bottom=.2,wspace=.15)
save(fig,'03_task')

# 4: distinguish existing design from the unimplemented continuous-shear idea.
fig,axs=plt.subplots(2,1,figsize=(10,5.8),sharex=True)
for ax,title,sy in zip(axs,['Current: synchronous final burst','Alternative: shear continues through final burst'],[True,False]):
    raster(ax,[-40,-20,0,20,40],final_sync=sy)
    ax.set_title(title,loc='left',pad=10); ax.set_xlabel('')
    ax.axvline(1010,color='#777777',lw=.8,ls=':')
axs[0].set_ylabel('Band centre (Hz)'); axs[-1].set_xlabel('Time (ms)')
fig.text(.12,.96,'Two different questions',fontsize=19)
fig.text(.12,.905,'Context before the target  /  Asynchrony during the target',fontsize=13)
fig.subplots_adjust(left=.12,right=.98,top=.79,bottom=.12,hspace=.55)
save(fig,'04_final_burst')
with PdfPages(OUT/'cmr_schematics.pdf') as pdf:
    for f in figs: pdf.savefig(f,bbox_inches='tight')
(OUT/'README.md').write_text('Schematic figures, not behavioral results.\n\nTiming source: tcmr/README.md and tcmr/stimulus.py, read 2026-11-12.\nFive categorical frequency rows: 250, 500, 1000, 2000, 4000 Hz. No logarithmic axes. Burst duration 187.5 ms, period 250 ms, four precursors. Ascending offsets [-40,-20,0,20,40] ms; zigzag [40,-20,0,20,-40] ms. Final burst synchronous in current code. Noise bars represent gates, not spectrogram bandwidth. Envelope panel is illustrative at zero shear; current code shifts entire noise tokens at nonzero shear. Alternative continuous shear is conceptual and is not implemented by this script. CMR formula requires a matched independent reference for the condition; existing preset does not supply all matched references. No inferred grouping claims or invented thresholds plotted.\n\nVisual specification: monochrome event rasters, envelope traces, and trial sequence; 10–11 inch landscape canvas; PNG/SVG and combined PDF; no uncertainty applies.\n')
print(OUT)
