import sys, importlib.util, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.argv=['x']
spec=importlib.util.spec_from_file_location('r5','/home/bisi/code/ibl-ai-agent/projects/ssl-whisker-artefact-gap/diagnosis/05_raw_after_5ms.py')
r5=importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
OUT='/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap/raw_after_5ms'
r,x,fs,uv,maxint,pt,on=r5.load_probe('MH020_20250304_161732','imec0')
# all active whisker trials this time
tr=r5.scr.read_trials('MH020_20250304_161732')
ft,a,b,*_=r5.scr.align(tr,r.whisker_times_file)
sel=tr[(tr.whisker_stim==1)&(tr.perf!=6)&(tr.context!='passive')]
tf=a*sel.start_time.to_numpy()+b; q=np.abs(ft[None]-tf[:,None]).argmin(1); tf=np.where(np.abs(ft[q]-tf)<0.005,ft[q],tf)
tms,raw,hp=r5.snippets(x,fs,uv,tf)
n0=np.searchsorted(tms,0)
bm=(tms>-45)&(tms<-5); pm=(tms>6)&(tms<18)
rb=np.sqrt((hp[:,bm]**2).mean(1)); rp=np.sqrt((hp[:,pm]**2).mean(1))
ratio=rp/np.median(rb,0)                    # trial x ch, relative to channel's median baseline
z=(tms>=0)&(tms<4)
satw=raw/uv; sat=((np.abs(satw)>=maxint-1)[:,z]).mean(1)   # trial x ch frac samples at ADC limit
peak=np.abs(raw[:,z]).max(1)                 # trial x ch artefact peak uV
# recovery time per trial (median over channels): first 1-ms bin >=5ms with rms ratio >=0.8 for 3 bins
e=np.arange(-60,81); bi=np.clip(np.searchsorted(e,tms,side='right')-1,0,len(e)-2); c=e[:-1]+.5
rt=[]
for k in range(len(hp)):
    rr=np.sqrt(np.array([ (hp[k,bi==j]**2).mean(0) for j in range(len(c))]))/np.median(rb,0)
    m=np.median(rr,1); ok=m>=0.8; t=np.nan
    for j in np.flatnonzero(c>=5):
        if ok[j:j+3].all(): t=c[j]-.5; break
    rt.append(t)
rt=np.array(rt)
print('n trials',len(hp),'trial time range s',tf.min().round(),tf.max().round())
print('recovery ms per trial: median',np.nanmedian(rt),'pct 10/90',np.nanpercentile(rt,[10,90]),'nan',np.isnan(rt).sum())
print('corr trial recovery vs time in session',np.corrcoef(np.nan_to_num(rt,nan=60),tf)[0,1])
print('corr trial recovery vs median artefact peak',np.corrcoef(np.nan_to_num(rt,nan=60),np.median(peak,1))[0,1])
cr=np.median(ratio,0); cs=sat.mean(0)
print('channel ratio 6-18 pct 10/50/90',np.percentile(cr,[10,50,90]).round(2))
print('corr channel ratio vs channel saturation frac',np.corrcoef(cr,cs)[0,1].round(2), 'vs channel index',np.corrcoef(cr,np.arange(384))[0,1].round(2))
for g,lo,hi in (('ch0-63',0,64),('ch64-191',64,192),('ch192-383',192,384)):
    print(g,'ratio',np.median(cr[lo:hi]).round(2),'sat frac',cs[lo:hi].mean().round(3),'baseline rms',np.median(np.median(rb,0)[lo:hi]).round(1))
fig,ax=plt.subplots(1,4,figsize=(20,6),gridspec_kw=dict(width_ratios=[3,3,1,1]))
o=np.argsort(tf)
im=ax[0].imshow(np.log2(ratio[o].T),aspect='auto',cmap='RdBu_r',vmin=-3,vmax=3,origin='lower'); plt.colorbar(im,ax=ax[0],label='log2 RMS 6-18 ms / channel baseline')
ax[0].set_xlabel('whisker trial (session order)'); ax[0].set_ylabel('channel'); ax[0].set_title('A  spike-band RMS 6-18 ms, trial x channel')
im=ax[1].imshow(sat[o].T,aspect='auto',cmap='Greys',vmin=0,vmax=1,origin='lower'); plt.colorbar(im,ax=ax[1],label='fraction 0-4 ms at ADC limit')
ax[1].set_xlabel('whisker trial (session order)'); ax[1].set_title('B  saturation 0-4 ms, trial x channel')
ax[2].plot(cr,np.arange(384),'k',lw=.7); ax[2].plot(cs,np.arange(384),'r',lw=.7); ax[2].set_title('C  channel: ratio (k), sat (r)'); ax[2].set_ylim(0,383)
ax[3].plot(np.nan_to_num(rt[o],nan=60),np.arange(len(o)),'.k'); ax[3].set_xlabel('recovery ms (60 = none)'); ax[3].set_ylabel('trial (session order)'); ax[3].set_title('D  per-trial recovery')
fig.suptitle(f'MH020_20250304 imec0: unevenness of the post-artefact gap across trials and channels (raw .ap.bin, {len(hp)} active whisker trials)')
fig.savefig(f'{OUT}/MH020_20250304_161732_imec0_trial_channel_map.png',dpi=120,bbox_inches='tight')
