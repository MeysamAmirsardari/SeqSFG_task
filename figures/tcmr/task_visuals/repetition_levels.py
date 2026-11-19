from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize,LinearSegmentedColormap
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tcmr.config import Config,Condition
from tcmr.stimulus import build
cfg=Config.load(ROOT/'tcmr/configs/tcmr_exp1.json');out=Path(__file__).parent
plt.rcParams.update({'font.family':'Helvetica Neue','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'svg.fonttype':'none','pdf.fonttype':42})
order=[-1,2,0,-2,1];levels=[]
for env in ['co','rand']:
 tr,_,stem=build(cfg,Condition(env,'frozen',40),65,[20261116,25],n_intervals=1,signal_interval=1,orders=[order],stems=True)
 rms=np.zeros((6,5))
 for j in range(6):
  for b in range(5):
   on=tr.target_on_s[0]-1+j*.2+(order[b]*.04 if j<5 else 0)
   k=round(on*cfg.sample_rate);sig=stem[b,k:k+2400]
   rms[j,b]=np.sqrt(np.mean(sig**2))
 levels.append(rms)
lev=20*np.log10(np.array(levels)/np.max(levels));lo=2*np.floor(lev.min()/2)
cm=LinearSegmentedColormap.from_list('burst_level',plt.get_cmap('plasma_r')(np.linspace(.08,.95,256)));norm=Normalize(lo,0)
f,axs=plt.subplots(5,2,figsize=(14,9),sharex=True,sharey=True)
f.subplots_adjust(left=.105,right=.975,bottom=.19,top=.84,hspace=.30,wspace=.17)
f.text(.105,.955,'Noise level varies across repetitions',fontsize=21)
for col,title in enumerate(['Comodulated','Independent']):
 axs[0,col].set_title(title,loc='left',fontsize=17,pad=25)
 for row,s in enumerate([0,5,10,20,40]):
  ax=axs[row,col]
  for j in range(6):
   for b in range(5):
    on=.08+j*.2+(order[b]*s/1000 if j<5 else 0)
    ax.add_patch(Rectangle((on,b-.31),.05,.62,fc=cm(norm(lev[col,j,b])),ec='none'))
  # The added target is separate from the masker RMS encoded by fill.
  ax.plot([1.08,1.13],[2,2],color='white',lw=4.5,solid_capstyle='butt')
  ax.plot([1.08,1.13],[2,2],color='black',lw=2,solid_capstyle='butt')
  ax.set_ylim(-.5,4.6);ax.set_xlim(-.01,1.15);ax.set_yticks([0,2,4],['250','1000','4000']);ax.tick_params(axis='y',length=0,labelsize=10);ax.spines['left'].set_visible(False);ax.set_xticks([0,.4,.8,1.1])
  if col==0:ax.text(-.13,2,f'{s} ms',ha='right',va='center',fontsize=14)
 axs[-1,col].set_xlabel('Time within interval (s)',labelpad=7)
f.text(.009,.51,'Band (Hz)',rotation=90,va='center',fontsize=12)
cax=f.add_axes([.105,.078,.40,.028]);cb=f.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal');cb.set_ticks(np.arange(lo,1,4));cb.outline.set_visible(False);cb.ax.tick_params(labelsize=11)
f.text(.105,.125,'Burst RMS level (dB relative to loudest burst)',fontsize=11)
f.text(.56,.105,'Colour: noise level',fontsize=12)
f.text(.56,.072,'Black line: added tone',fontsize=12)
f.text(.105,.022,'50 ms bursts  ·  200 ms period  ·  Final burst always synchronous\nGenerated examples; the same noise tokens are reused across rows to isolate the timing change.',fontsize=11,linespacing=1.5)
for ext in ['png','svg','pdf']:f.savefig(out/f'08_repetition_levels_colour.{ext}',dpi=180,facecolor='white')
np.savez(out/'08_repetition_levels_colour_data.npz',rms=np.array(levels),relative_db=lev,seed=[20261116,25],order=order)
print('Relative level range:',lev.min(),lev.max())
