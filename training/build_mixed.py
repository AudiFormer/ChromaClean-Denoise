"""Merge clean training pictures into one RGB crop set for train_cf.py, rejecting low-quality crops.

Sources:
  - raw.pixls.us CC0 pictures (rpu/pics/*.u16, 3 x h x w cell-binned camera RGB from `rawdump bin`,
    65535 = clipped cell), many cameras;
  - the author's own crops (clean_crops.npy, packed RGGB), converted to binned RGB.

Quality gate, per crop (measured on the green plane):
  - noise: the leftover noise of the "clean" picture, as a shot-noise coefficient a = var / signal, read from
    the flattest blocks (a 4-neighbour Laplacian, 20 sigma^2 for white noise). Too noisy = not clean.
  - blur: fine-detail energy (Laplacian) over mid-detail energy (difference of blurs). A textured crop
    with little fine detail is out of focus or shaken.
  - darkness and clipping.
Pass 1 measures every candidate; cut-offs are set from the distribution (`--stats` prints it); pass 2
writes the kept crops: clean_rgb.npy (N,3,256,256) float16 (clipped cells > 1) and clean_rgb.json with a
source tag and sampling weight per crop, so each source gets a set share of the batches.

Usage: python build_mixed.py [--stats]"""
import csv, json, os, random, sys
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

from common import WORK as HERE, ROOT, CODE  # noqa: E402  data/work folder (see common.py)
RPU = f'{ROOT}/rpu'
S = 256
PER_PIC = 12
SHARE = {'rpu': 0.45, 'cube': 0.12, 'sony': 0.28, 'olympus': 0.15}
CUBE = f'{ROOT}/cube'
# cut-offs (set from the measured distributions; see --stats)
MIN_MEDIAN = 0.01          # median brightness of the crop
MAX_CLIP = 0.05            # fraction of clipped cells
NOISE_PCT = 90             # reject crops noisier than this percentile of all candidates ...
NOISE_ABS = 3e-4           # ... or than this shot-noise coefficient (about ISO 800 on full frame), whichever is lower
BLUR_PCT = 12              # among textured crops, reject the least sharp this percentile


def metrics(c):
    """c: (3,S,S) float RGB, white 1 (clipped > 1). Returns dict of quality figures."""
    g = c[1].astype(np.float64)
    clip = float((c.max(0) >= 0.99).mean())
    med = float(np.median(g))
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    # flattest blocks give the noise
    B = 16
    h, w = (lap.shape[0] // B) * B, (lap.shape[1] // B) * B
    lv = (lap[:h, :w] ** 2).reshape(h // B, B, w // B, B).mean((1, 3)) / 20.0
    mean_b = g[1:1 + h, 1:1 + w].reshape(h // B, B, w // B, B).mean((1, 3))
    dog = np.abs(gaussian_filter(g, 1.5) - gaussian_filter(g, 4.0))
    md_b = dog[1:1 + h, 1:1 + w].reshape(h // B, B, w // B, B).mean((1, 3))
    # a flat block: its mid-scale detail is small next to its fine-scale (noise) level. Pure noise gives a
    # ratio of about 0.12; texture gives 0.5 and up. Only flat blocks can tell noise from detail.
    flat = (mean_b > 0.01) & (mean_b < 0.9) & (md_b < 0.35 * np.sqrt(lv))
    if flat.sum() >= 4:
        noise = float(np.median(lv[flat] / mean_b[flat]))
    else:
        noise = 0.0          # no flat area: noise cannot be told from detail, so it is not judged
    hi = float(np.abs(lap).mean())
    mid = float(dog.mean())
    texture = mid / max(float(g.mean()), 1e-3)
    sharp = hi / max(mid, 1e-6)
    return {'clip': clip, 'median': med, 'noise': noise, 'texture': texture, 'sharp': sharp}


def own_candidates():
    meta = json.load(open(f'{HERE}/clean_crops.json'))
    own = np.load(f'{HERE}/clean_crops.npy', mmap_mode='r')
    tags = []
    for path, k in zip(meta['images'], meta['per_image']):
        tags += [('sony' if path.lower().endswith('.arw') else 'olympus', os.path.basename(path))] * k
    for i in range(meta['count']):
        p = np.asarray(own[i]).astype(np.float32)
        yield ('own', i), tags[i][0], tags[i][1], np.stack([p[0], 0.5 * (p[1] + p[2]), p[3]])


def rpu_candidates():
    if not os.path.exists(f'{RPU}/sources.csv'):
        return
    rows = list(csv.DictReader(open(f'{RPU}/sources.csv', encoding='utf8')))
    for r in rows:
        w, h = int(r['width']), int(r['height'])
        p = f'{RPU}/pics/{r["file"]}.u16'
        if not os.path.exists(p) or w < S + 8 or h < S + 8:
            continue
        a = np.memmap(p, dtype='<u2', mode='r', shape=(3, h, w))
        rnd = random.Random(r['sha256'])
        for k in range(PER_PIC * 3):          # more positions than needed: the gate drops some
            y, x = rnd.randrange(4, h - S - 4), rnd.randrange(4, w - S - 4)
            yield ('rpu', r['file'], y, x), 'rpu', f'{r["make"]} {r["model"]}', np.asarray(a[:, y:y + S, x:x + S]).astype(np.float32) / 65534.0


def cube_candidates():
    """Cube++ (CC BY 4.0) crops already cut by cube_fetch.py: u16, 65535 = clipped."""
    if not os.path.exists(f'{CUBE}/sources.csv'):
        return
    import glob
    for path in sorted(glob.glob(f'{CUBE}/crops_*.npy')):
        k = int(os.path.basename(path)[6:-4])
        a = np.load(path, mmap_mode='r')
        for i in range(len(a)):
            yield ('cube', k, i), 'cube', 'Canon EOS 550D/600D', np.asarray(a[i]).astype(np.float32) / 65534.0


def main():
    stats = '--stats' in sys.argv
    cand = []
    for gen in (own_candidates(), rpu_candidates(), cube_candidates()):
        for key, src, cam, c in gen:
            m = metrics(c); m.update(key=key, src=src, cam=cam)
            cand.append(m)
            if len(cand) % 2000 == 0:
                print(len(cand), 'measured', flush=True)
    noise_all = np.array([m['noise'] for m in cand if np.isfinite(m['noise']) and m['noise'] > 0])
    noise_cut = min(float(np.percentile(noise_all, NOISE_PCT)), NOISE_ABS)
    textured = np.array([m['sharp'] for m in cand if m['texture'] > np.median([x['texture'] for x in cand])])
    blur_cut = float(np.percentile(textured, BLUR_PCT))
    tex_med = float(np.median([x['texture'] for x in cand]))
    for src in SHARE:
        ms = [m for m in cand if m['src'] == src]
        if not ms:
            continue
        nz = np.array([m['noise'] for m in ms if np.isfinite(m['noise']) and m['noise'] > 0]); nz = nz if len(nz) else np.zeros(1)
        sh = np.array([m['sharp'] for m in ms])
        print(f'{src:8} n={len(ms):6}  noise a p50 {np.median(nz):.2e} p90 {np.percentile(nz, 90):.2e}  '
              f'sharp p10 {np.percentile(sh, 10):.2f} p50 {np.median(sh):.2f}  median<{MIN_MEDIAN}: '
              f'{np.mean([m["median"] < MIN_MEDIAN for m in ms]):.1%}  clip>{MAX_CLIP}: {np.mean([m["clip"] > MAX_CLIP for m in ms]):.1%}')
    print(f'cut-offs: noise a > {noise_cut:.2e}, sharpness < {blur_cut:.2f} on crops with texture > {tex_med:.4f}')

    def keep(m):
        if m['clip'] > MAX_CLIP or m['median'] < MIN_MEDIAN or not np.isfinite(m['noise']):
            return 'dark/clipped/unmeasurable'
        if m['noise'] > noise_cut:
            return 'noisy'
        if m['texture'] > tex_med and m['sharp'] < blur_cut:
            return 'blurry'
        return None
    why = [keep(m) for m in cand]
    # cap CC0 crops per picture after the gate
    per, kept = {}, []
    for m, w in zip(cand, why):
        if w:
            continue
        if m['src'] == 'rpu':
            k = m['key'][1]
            if per.get(k, 0) >= PER_PIC:
                continue
            per[k] = per.get(k, 0) + 1
        kept.append(m)
    for src in SHARE:
        tot = sum(m['src'] == src for m in cand)
        reasons = {r: sum(1 for m, w in zip(cand, why) if m['src'] == src and w == r) for r in ('dark/clipped/unmeasurable', 'noisy', 'blurry')}
        if not tot:
            continue
        print(f'{src:8} candidates {tot:6}  rejected {reasons}  kept {sum(m["src"] == src for m in kept)}')
    # the worst cameras by rejection, for the record
    if stats:
        return
    N = len(kept)
    mm = np.lib.format.open_memmap(f'{HERE}/clean_rgb.npy', mode='w+', dtype=np.float16, shape=(N, 3, S, S))
    own = np.load(f'{HERE}/clean_crops.npy', mmap_mode='r')
    rows = {r['file']: r for r in csv.DictReader(open(f'{RPU}/sources.csv', encoding='utf8'))}
    for j, m in enumerate(kept):
        k = m['key']
        if k[0] == 'cube':
            c = np.asarray(np.load(f'{CUBE}/crops_{k[1]}.npy', mmap_mode='r')[k[2]]).astype(np.float32) / 65534.0
        elif k[0] == 'own':
            p = np.asarray(own[k[1]]).astype(np.float32)
            c = np.stack([p[0], 0.5 * (p[1] + p[2]), p[3]])
        else:
            r = rows[k[1]]
            a = np.memmap(f'{RPU}/pics/{k[1]}.u16', dtype='<u2', mode='r', shape=(3, int(r['height']), int(r['width'])))
            c = np.asarray(a[:, k[2]:k[2] + S, k[3]:k[3] + S]).astype(np.float32) / 65534.0
        mm[j] = c.astype(np.float16)
    mm.flush(); del mm
    src = [m['src'] for m in kept]
    count = {k: src.count(k) for k in SHARE}
    weights = [SHARE[s] / max(count[s], 1) for s in src]
    json.dump({'count': N, 'size': S, 'source': src, 'camera': [m['cam'] for m in kept], 'weight': weights,
               'counts': count, 'cutoffs': {'noise_a': noise_cut, 'sharp': blur_cut, 'texture_median': tex_med,
                                             'min_median': MIN_MEDIAN, 'max_clip': MAX_CLIP}},
              open(f'{HERE}/clean_rgb.json', 'w'))
    print('wrote', N, 'crops', count)


if __name__ == '__main__':
    main()
