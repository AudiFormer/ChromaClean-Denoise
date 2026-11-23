"""Measure row/column (banding) noise from dark frames and add it to the profile."""
import json, os
import numpy as np
from common import load_packed, exif, ROOT
d=f'{ROOT}/sony/calib_dark'; pf=f'{ROOT}/sony/profile_sony_a7iv.json'
prof=json.load(open(pf))
by={}
for f in sorted(os.listdir(d)): by.setdefault(exif(f'{d}/{f}')['iso'],[]).append(f'{d}/{f}')
print('ISO | pixel std DN | row std DN (ratio to pixel) | col std DN | row corr across channels')
for iso in sorted(by):
    fr=[load_packed(p)[0] for p in by[iso][:6]]
    mean=np.mean(fr,axis=0)                      # fixed pattern (stays frame to frame)
    rs,cs,ps,cc=[],[],[],[]
    for x in fr[:4]:
        e=x-mean                                 # temporally varying part only
        ps.append(e.std())
        rm=e.mean(2); cm=e.mean(1)               # per channel row / column means
        rs.append(np.sqrt(max(rm.var()-e.var()/e.shape[2],0))); cs.append(np.sqrt(max(cm.var()-e.var()/e.shape[1],0)))
        cc.append(np.corrcoef(rm)[np.triu_indices(4,1)].mean())
    k=6/5  # residual from subtracting a 6-frame mean
    p,r,c=np.mean(ps)*np.sqrt(k),np.mean(rs)*np.sqrt(k),np.mean(cs)*np.sqrt(k)
    fpr=np.sqrt(max(mean.mean(2).var()-p**2/6/mean.shape[2],0))
    print(f'{iso:>6} | {p:7.2f} | {r:6.3f} ({r/p:.3f}) | {c:6.3f} | {np.mean(cc):+.2f} | fixed row {fpr:.3f}')
    prof['iso'][str(iso)]['row_noise_dn']=round(float(r),4); prof['iso'][str(iso)]['row_corr']=round(float(np.mean(cc)),3)
json.dump(prof,open(pf,'w'),indent=1); print('profile updated')
