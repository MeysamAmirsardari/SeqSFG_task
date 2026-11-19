from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tcmr.config import Config,Condition
from tcmr.stimulus import build,envelope
out=Path(__file__).parent;cfg=Config.load(ROOT/'tcmr/configs/tcmr_exp1.json')
plt.rcParams.update({'font.family':'Helvetica Neue','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42})
levels=[-12,-6,0];colors=['#c5aebd','#96677f','#63304d'];widths=[1.2,2.7,5.0]
figs=[]
def save(f,n):
 figs.append(f)
 for ext in ['png','svg','pdf']:f.savefig(out/f'{n}.{ext}',dpi=180,facecolor='white')
# Identical timing and masker for an explanatory contrast, not three trials from a recording.
tr,_,stems=build(cfg,Condition('co','frozen',20),60,[20261116,7],n_intervals=1,signal_interval=1,orders=[[-1,2,0,-2,1]],stems=True)
f,axs=plt.subplots(3,1,figsize=(11,6.7),sharex=True);f.subplots_adjust(left=.13,right=.97,bottom=.21,top=.84,hspace=.48)
f.text(.13,.95,'Only the added tone changes level',fontsize=20)
f.text(.13,.89,'Grey: noise     Plum: target tone',fontsize=13)
for i,(ax,lev,col,w) in enumerate(zip(axs,levels,colors,widths)):
 t0=tr.target_on_s[0]-1-tr.interval_on_s[0]
 for j in range(6):
  for b in range(5):
   on=t0+j*.2+(tr.slots[0][j][b]*.02 if j<5 else 0)
   ax.add_patch(Rectangle((on,b-.17),.05,.34,fc='.79',ec='none'))
 on=tr.target_on_s[0]-tr.interval_on_s[0]
 ax.plot([on,on+.05],[2,2],color=col,lw=w,solid_capstyle='butt')
 ax.text(.005,4.8,f'{lev:d} dB',fontsize=15,va='bottom')
 ax.set_ylim(-.5,5.3);ax.set_xlim(0,1.15);ax.set_yticks([0,2,4],['250','1000','4000']);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False)
axs[1].set_ylabel('Band (Hz)');axs[-1].set_xlabel('Time within interval (s)')
f.text(.13,.035,'Illustration: same masker and timing. Target levels relative to the strongest example.\nDarker, thicker stroke = stronger target; stroke thickness does not represent bandwidth.',fontsize=11,linespacing=1.5)
save(f,'05_level_in_elements')
# True signal amplitude: samples from the stimulus generator, using identical seed and phase.
f,axs=plt.subplots(1,3,figsize=(11,3.8),sharex=True,sharey=True);f.subplots_adjust(left=.1,right=.98,bottom=.24,top=.73,wspace=.18)
f.text(.1,.94,'The target waveform gets smaller',fontsize=20)
ref=None
for ax,lev,col in zip(axs,levels,colors):
 t,_,s=build(cfg,Condition('co','frozen',20),60+lev,[20261116,7],n_intervals=1,signal_interval=1,orders=[[-1,2,0,-2,1]],stems=True)
 start=round(t.target_on_s[0]*cfg.sample_rate);tone=s[-1,start:start+2400]
 if ref is None:ref=cfg.uncal_peak_60db
 tt=np.arange(len(tone))/cfg.sample_rate*1000
 ax.plot(tt,tone/ref,c=col,lw=.8)
 ax.set_title(f'{lev:d} dB',fontsize=15,pad=12);ax.set_xlim(0,50);ax.set_ylim(-1.1,1.1);ax.set_xticks([0,25,50]);ax.set_yticks([-1,0,1]);ax.set_xlabel('Time (ms)')
axs[0].set_ylabel('Relative amplitude')
f.text(.1,.055,'1 kHz tone  ·  50 ms  ·  Same phase and ramps  ·  Levels relative to the strongest example',fontsize=11)
save(f,'06_target_waveforms')
with PdfPages(out/'target_level.pdf') as p:
 for f in figs:p.savefig(f)
print('Saved level diagrams')
