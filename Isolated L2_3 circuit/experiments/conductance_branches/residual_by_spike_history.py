"""Residual of the frozen front end vs the input the Izhikevich layer needed (near_threshold.py
inversion), binned by time since the last spike and by Vm. Run from "Isolated L2_3 circuit/"."""
import sys, json, numpy as np, tensorflow as tf
sys.path.insert(0,'experiments/conductance_branches')
import near_threshold as nt
sb, sp, fs = nt.sb, nt.sp, nt.fs
sets = fs.load_split()
m,_ = sp.load_frozen("dense3")
r = json.load(open("experiments/conductance_branches/results_fast_k/dense3/metrics.json"))
cell = sb.BranchCell(r["frozen"], r["trained"]); cell.I0 = float(r["frozen"]["I0"]); g=float(r["frozen"]["g"])
for pr in ("noise","ramp","square"):
    s = sets[(pr,"val")]; I=s["I"].astype(np.float32); V=s["V"].astype(np.float32)
    x = m.current_np(I).astype(np.float32)
    tgt, w = nt.clamp_targets(cell, x, I, V)
    res = (x - g*(I+cell.I0)) - tgt     # model minus needed: + means model pushes up too much
    S,_ = sp.gt_masks(V)
    tss = np.full(V.shape, 9999.0)
    for i in range(len(V)):
        last=-9999
        for t in range(V.shape[1]):
            if S[i,t]>0.5: last=t
            tss[i,t]=t-last
    k = w>0
    print(pr, "residual (x units; 1 unit = %.1f pA)" % (1/g))
    for lo,hi in ((3,10),(10,25),(25,50),(50,100),(100,200),(200,500),(500,2000),(2000,1e9)):
        sel = k & (tss>=lo) & (tss<hi)
        if sel.sum()>100: print("   since spike %5d-%-5s ms: mean %+.2f  rms %.2f  n %d" % (lo, int(hi) if hi<1e8 else 'never', res[sel].mean(), np.sqrt((res[sel]**2).mean()), sel.sum()))
    for lo,hi in ((-90,-60),(-60,-50),(-50,-45),(-45,-40)):
        sel = k & (V>=lo)&(V<hi)
        if sel.sum()>100: print("   V %d..%d: mean %+.2f rms %.2f" % (lo,hi,res[sel].mean(), np.sqrt((res[sel]**2).mean())))
