"""Use the downloaded beginning of an unfinished Cube++ zip: join its leading run of complete chunks, salvage the
CR2 files whose CRC checks out, and process them like a full zip (crops_<k>.npy, sources.csv, done_<k>)."""
import csv, glob, os, shutil, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cube_fetch as C

for k in map(int, sys.argv[1:]):
    z = f'{C.OUT}/source_CR2_{k}.zip'
    size = C.sizes()[f'source_CR2_{k}.zip']
    n = 0
    with open(z + '.partial', 'wb') as out:
        while True:
            p = f'{z}.part{n:04d}'
            want = min(C.CHUNK, size - n * C.CHUNK)
            if not os.path.exists(p) or os.path.getsize(p) != want:
                break
            with open(p, 'rb') as t:
                shutil.copyfileobj(t, out, 1 << 20)
            n += 1
    print(f'zip {k}: first {n} chunks ({n * C.CHUNK / 1e9:.1f} of {size / 1e9:.1f} GB)', flush=True)
    ex = f'{C.OUT}/x_{k}'; os.makedirs(ex, exist_ok=True)
    good, bad = C.salvage(z + '.partial', ex)
    print(f'zip {k}: {good} intact CR2 files ({bad} damaged spots; the cut-off last file counts as one)', flush=True)
    files = sorted({os.path.normcase(f): f for f in glob.glob(f'{ex}/*.CR2') + glob.glob(f'{ex}/*.cr2')}.values())
    kept, rows, skipped = [], [], {}
    with ThreadPoolExecutor(6) as pool:
        for cr2, res, why in pool.map(lambda f: C.crops_of(f, ex), files):
            if why:
                key = why.split(':')[0].split(' ')[0]; skipped[key] = skipped.get(key, 0) + 1
            elif res and res[1] is not None:
                meta, c = res
                kept.append(c)
                rows.append([os.path.basename(cr2), meta.get('make'), meta.get('model'), meta.get('iso'), len(c),
                             'CC-BY-4.0', 'https://zenodo.org/records/4153431'])
    if kept:
        np.save(f'{C.OUT}/crops_{k}.npy', np.concatenate(kept))
    with open(f'{C.OUT}/sources.csv', 'a', newline='', encoding='utf8') as f:
        csv.writer(f).writerows(rows)
    shutil.rmtree(ex, ignore_errors=True)
    for p in glob.glob(z + '.part*'):
        os.remove(p)
    open(f'{C.OUT}/done_{k}', 'w').write('partial')
    print(f'zip {k}: kept {len(rows)} pictures ({sum(len(c) for c in kept)} crops), skipped {skipped}', flush=True)
