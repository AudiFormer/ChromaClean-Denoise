"""Sample sheet of what the quality gate rejects (noisy, blurry) and keeps, from the author's own crops."""
import numpy as np, random
from PIL import Image
import build_mixed as B
cand=[]
for key,src,cam,c in B.own_candidates():
    m=B.metrics(c); cand.append((m,c))
    if len(cand)>=6000: break
noise=np.array([m['noise'] for m,_ in cand if m['noise']>0]); ncut=min(np.percentile(noise,B.NOISE_PCT),B.NOISE_ABS)
tex_med=np.median([m['texture'] for m,_ in cand]); sh=np.array([m['sharp'] for m,_ in cand if m['texture']>tex_med]); bcut=np.percentile(sh,B.BLUR_PCT)
groups={'noisy':[], 'blurry':[], 'kept':[]}
for m,c in cand:
    if m['clip']>B.MAX_CLIP or m['median']<B.MIN_MEDIAN: continue
    g='noisy' if m['noise']>ncut else 'blurry' if (m['texture']>tex_med and m['sharp']<bcut) else 'kept'
    groups[g].append(c)
print({k:len(v) for k,v in groups.items()}, 'judged for noise:', f'{np.mean([m["noise"]>0 for m,_ in cand]):.0%}')
rnd=random.Random(3)
def show(c):
    c=c[:,64:192,64:192]; g=c[1].mean(); wb=[g/max(c[0].mean(),1e-4),1,g/max(c[2].mean(),1e-4)]
    x=np.clip(np.stack([c[0]*wb[0],c[1],c[2]*wb[2]],-1)/max(np.percentile(c[1],99),1e-3),0,1)**(1/2.2)
    return (x*255).astype(np.uint8)
rows=[np.concatenate([show(c) for c in rnd.sample(groups[k],min(8,len(groups[k])))],1) for k in ('noisy','blurry','kept')]
Image.fromarray(np.concatenate(rows,0)).resize((8*160,3*160),Image.NEAREST).save('gate_samples.png')
