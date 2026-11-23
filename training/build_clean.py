"""Pick clean low-ISO RAWs from the inventory and cut packed crops into one memmap.

Output: clean_crops.f16 (N,4,S,S) normalised 0..1 (DN/white_range), clean_crops.json."""
import csv, json, os, random, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from common import load_packed

from common import WORK as HERE, ROOT, CODE  # noqa: E402  data/work folder (see common.py)
S = 256          # packed crop size (= 512x512 sensor pixels)
PER_IMAGE = 12
N_SONY = 900

def pick():
    rows = list(csv.DictReader(open(f'{HERE}/inventory.csv', encoding='utf-8')))
    sony = [r for r in rows if r['model'] == 'ILCE-7M4' and r['iso'].isdigit() and int(r['iso']) <= 100]
    oly = [r for r in rows if r['model'] == 'E-M10MarkIV' and r['iso'] == '100']
    # spread Sony picks across folders (round robin) to avoid bursts of one subject
    by = {}
    for r in sony:
        by.setdefault(os.path.dirname(r['path']), []).append(r)
    rnd = random.Random(0)
    for v in by.values():
        rnd.shuffle(v)
    picks, lists = [], list(by.values())
    while any(lists):
        for v in lists:
            if v:
                picks.append(v.pop())
    return [r['path'] for r in picks] + [r['path'] for r in oly]

def crops(path):
    try:
        x, rng = load_packed(path)
    except Exception as e:
        return path, None
    x = x / rng
    _, H, W = x.shape
    rnd = random.Random(hash(path) & 0xffffffff)
    out, tries = [], 0
    while len(out) < PER_IMAGE and tries < PER_IMAGE * 8:
        tries += 1
        y0 = rnd.randrange(0, H - S); x0 = rnd.randrange(0, W - S)
        c = x[:, y0:y0 + S, x0:x0 + S]
        m = c.mean()
        if m < 0.01 or (c >= 0.98).mean() > 0.3:   # skip near-black and mostly-clipped crops
            continue
        out.append(c.astype(np.float16))
    return path, (np.stack(out) if out else None)

if __name__ == '__main__':
    paths = pick()
    print(len(paths), 'images', flush=True)
    cap = (N_SONY + 200) * PER_IMAGE
    mm = np.lib.format.open_memmap(f'{HERE}/clean_crops.npy', mode='w+', dtype=np.float16, shape=(cap, 4, S, S))
    n, used, n_sony, per = 0, [], 0, []
    with ProcessPoolExecutor(8) as ex:
        for i, (p, c) in enumerate(ex.map(crops, paths, chunksize=2)):
            is_sony = p.lower().endswith('.arw')
            if c is not None and not (is_sony and n_sony >= N_SONY):
                mm[n:n + len(c)] = c; n += len(c); used.append(p); n_sony += is_sony; per.append(len(c))
            if i % 50 == 0:
                print(f'{i}/{len(paths)} images, {n} crops', flush=True)
    mm.flush(); del mm
    print('sony images used', n_sony)
    json.dump({'count': n, 'size': S, 'images': used, 'per_image': per}, open(f'{HERE}/clean_crops.json', 'w'), indent=1)
    print('done', n, 'crops from', len(used), 'images')
