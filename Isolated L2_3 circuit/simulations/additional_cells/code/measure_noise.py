"""Measure baseline (zero-command) Vm noise from Allen current-clamp NWB files.

Usage: python measure_noise.py <dir with DANDI 000020 *_icephys.nwb> ../noise_model/noise_psd.npz
"""
import h5py, numpy as np, sys, glob, json
from scipy import signal
NPER=512  # ms -> ~1.95 Hz resolution
out=[]
for fn in sorted(glob.glob(sys.argv[1]+'/*.nwb')):
    f=h5py.File(fn,'r'); segs=[]; segs_full=[]
    for k in f['acquisition']:
        g=f['acquisition'][k]
        if g.attrs['neurodata_type']!='CurrentClampSeries': continue
        rate=float(g['starting_time'].attrs['rate'])
        if rate!=50000: continue
        v=g['data'][()]*g['data'].attrs['conversion']*1e3
        s=f['stimulus/presentation'][k.replace('AD','DA')]
        st=s['data'][()]*s['data'].attrs['conversion']*1e12
        n=min(len(v),len(st)); v=v[:n]; st=st[:n]
        good=np.isfinite(v)&(v!=0)
        z=(np.abs(st)<1e-3)&good
        # mask 300 ms after any command change (and after the test pulse)
        ch=np.flatnonzero(np.diff((np.abs(st)>1e-3).astype(int))!=0)
        for c in ch: z[c:c+int(0.3*rate)]=False
        # contiguous runs
        d=np.diff(np.r_[0,z.astype(int),0]); a=np.flatnonzero(d==1); b=np.flatnonzero(d==-1)
        for i0,i1 in zip(a,b):
            if (i1-i0)/rate<NPER/1000+0.01: continue
            seg=v[i0:i1]
            if seg.max()>-30 or seg.min()<-110: continue   # spikes / broken
            segs_full.append(seg)
            segs.append(seg[::50])  # point-sample every 1 ms, like the simulations
    if not segs: print(fn,'no segs'); continue
    P=[];rms=[];rmsf=[];hum=[]
    for s,sf in zip(segs,segs_full):
        x=signal.detrend(s,type='constant')
        fr,p=signal.welch(x,fs=1000,nperseg=NPER,detrend='linear',window='hann')
        P.append(p); rms.append(np.std(signal.detrend(s))); rmsf.append(np.std(signal.detrend(sf)))
        # 60 Hz line amplitude via lock-in on full-rate data
        t=np.arange(len(sf))/50000.; xf=signal.detrend(sf)
        hum.append(2*np.abs(np.mean(xf*np.exp(-2j*np.pi*60*t))))
    P=np.median(np.array(P),0)
    name=fn.split('/')[-1].replace('_icephys.nwb','')
    rec=dict(session=name,n_segments=len(segs),total_s=float(sum(len(s) for s in segs)/1000),
             rms_1kHz_mV=float(np.median(rms)),rms_50kHz_mV=float(np.median(rmsf)),hum60_amp_mV=float(np.median(hum)),
             vrest_mV=float(np.median([np.median(s) for s in segs])))
    print(json.dumps(rec)); out.append((rec,P))
np.savez(sys.argv[2],freq=fr,psd=np.array([p for _,p in out]),sessions=np.array([r['session'] for r,_ in out]))
json.dump([r for r,_ in out],open(sys.argv[2].replace('.npz','.json'),'w'),indent=1)
