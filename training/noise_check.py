"""Compare real noise (noisy - aligned reference) against the calibrated model.

Reports per ISO: measured/predicted variance across brightness bins, residual
neighbour correlation (0 = white noise, >0 = smoothed or misaligned), and row-noise."""
import json, os, sys
import numpy as np
from scipy.ndimage import uniform_filter
from common import load_profile, noise_params

from common import WORK as HERE, ROOT, CODE  # noqa: E402  data/work folder (see common.py)
scene = sys.argv[1] if len(sys.argv) > 1 else 'scene_001'
d = f'{HERE}/pairs/{scene}'
idx = json.load(open(f'{d}/index.json'))
prof = load_profile(sys.argv[2]) if len(sys.argv) > 2 else load_profile(); R = idx['white_range']; nb = len(idx['base_frames'])
a0, b0 = noise_params(prof, idx['base_iso'], R)
seen = set()
print('ISO   | var ratio (measured/predicted) by brightness: dark -> bright | neighbour corr x,y | row-noise ratio')
for fr in idx['frames']:
    iso = fr['iso']
    if iso in seen:
        continue
    seen.add(iso)
    n = np.load(f'{d}/noisy_{fr["file"][:-4]}.npy').astype(np.float32)
    r = np.load(f'{d}/ref_{fr["file"][:-4]}.npy').astype(np.float32)
    a, b = noise_params(prof, iso, R)
    res = n - r
    ratios, cx, cy, rows = [], [], [], []
    edges = np.geomspace(0.003, 0.6, 8)
    acc = {i: [0.0, 0.0] for i in range(len(edges) - 1)}
    for c in range(4):
        x = np.clip(r[c], 0, None)
        pred = a * x + b[c] + (a0 * x + b0[c]) / nb
        # smooth areas only: local contrast of the reference small vs the noise
        loc = np.sqrt(np.clip(uniform_filter(r[c] ** 2, 5) - uniform_filter(r[c], 5) ** 2, 0, None))
        smooth = loc < 3 * np.sqrt((a0 * x + b0[c]) / nb) + 0.002
        for i in range(len(edges) - 1):
            m = smooth & (x >= edges[i]) & (x < edges[i + 1])
            if m.sum() > 2000:
                acc[i][0] += (res[c][m] ** 2).mean(); acc[i][1] += pred[m].mean()
        e = res[c] / np.sqrt(pred)
        sm = smooth[:, 1:] & smooth[:, :-1]
        cx.append(np.corrcoef(e[:, 1:][sm], e[:, :-1][sm])[0, 1])
        sm = smooth[1:, :] & smooth[:-1, :]
        cy.append(np.corrcoef(e[1:, :][sm], e[:-1, :][sm])[0, 1])
        rm = e.mean(1); rows.append(rm.std() * np.sqrt(e.shape[1]))
    rat = ' '.join(f'{acc[i][0] / acc[i][1]:4.2f}' if acc[i][1] else '  - ' for i in acc)
    print(f'{iso:>5} | {rat} | {np.mean(cx):+.3f},{np.mean(cy):+.3f} | {np.mean(rows):5.2f}', flush=True)
