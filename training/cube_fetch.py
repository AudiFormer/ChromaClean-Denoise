"""Fetch the Cube++ dataset (CC BY 4.0, Canon 550D/600D CR2 raws) zip by zip and keep only clean training crops.

Per zip: download (resumable), extract, turn each CR2 into cell-binned camera RGB with the app's own decoder
(`rawdump bin`), skip ISO > 400, cut CROPS random 256-cell crops that are not mostly clipped or black, store them
as u16 (65535 = clipped cell) in cube/crops_<k>.npy, record the source in cube/sources.csv, then delete the raws,
the binned picture and the zip. Peak disk use is about one zip plus its extracted files.

Credit (CC BY 4.0): "The Cube++ Illumination Estimation Dataset", E. Ershov, A. Savchik, I. Semenkov, N. Banic,
A. Belokopytov, D. Senshina, K. Koscevic, M. Subasic, S. Loncaric, IEEE Access 2020, https://zenodo.org/records/4153431

Usage: python cube_fetch.py [first_zip last_zip]"""
import csv, glob, json, os, random, shutil, subprocess, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from common import WORK as HERE, ROOT, CODE  # noqa: E402  data/work folder (see common.py)
OUT = f'{ROOT}/cube'
RAWDUMP = f'{HERE}/rawdump-target/release/lc-denoise-rawdump.exe'
SOURCES = {
    # name: (url pattern, parallel connections); both serve the same files, each server caps its total speed
    'zenodo': ('https://zenodo.org/api/records/4153431/files/source_CR2_{k}.zip/content', 6),
    'zenodo2': ('https://zenodo.org/api/records/4153431/files/source_CR2_{k}.zip/content', 6),  # FTP mirror dropped: its ranged reads came back corrupted
}
CHUNK = 32 * 1024 * 1024
SIZES = {}


def sizes():
    if not SIZES:
        import urllib.request
        d = json.load(urllib.request.urlopen('https://zenodo.org/api/records/4153431', timeout=60))
        for f in d['files']:
            SIZES[f['key']] = f['size']
    return SIZES


def download(k, src, z):
    """Ranged, parallel, resumable download into chunk files, then joined."""
    url, conns = SOURCES[src]
    url = url.format(k=k)
    size = sizes()[f'source_CR2_{k}.zip']
    parts = [(i, o, min(o + CHUNK, size) - 1) for i, o in enumerate(range(0, size, CHUNK))]
    def get(p):
        # each chunk is fetched whole into a temp file and kept only if complete: never stitched from pieces
        i, a, b = p
        f = f'{z}.part{i:04d}'
        for attempt in range(40):
            if os.path.exists(f) and os.path.getsize(f) == b - a + 1:
                return True
            for x in (f, f + '.tmp'):
                if os.path.exists(x):
                    os.remove(x)
            subprocess.run(['curl', '-sS', '-f', '-L', '--max-time', '900', '-r', f'{a}-{b}', '-o', f + '.tmp', url])
            if os.path.exists(f + '.tmp') and os.path.getsize(f + '.tmp') == b - a + 1:
                os.replace(f + '.tmp', f)
        return os.path.exists(f) and os.path.getsize(f) == b - a + 1
    with ThreadPoolExecutor(conns) as pool:
        ok = all(pool.map(get, parts))
    if not ok:
        return False
    with open(z, 'wb') as out:
        for i, _, _ in parts:
            with open(f'{z}.part{i:04d}', 'rb') as t:
                shutil.copyfileobj(t, out, 1 << 20)
            os.remove(f'{z}.part{i:04d}')
    return os.path.getsize(z) == size
S, CROPS, MAX_ISO = 256, 3, 400


def crops_of(cr2, tmpdir):
    prefix = os.path.join(tmpdir, os.path.splitext(os.path.basename(cr2))[0])
    p = subprocess.run([RAWDUMP, 'bin', cr2, prefix], capture_output=True, text=True, timeout=600)
    if p.returncode:
        return cr2, None, 'decode: ' + p.stderr.strip()[:120]
    try:
        meta = json.loads(p.stdout)
        if not os.path.exists(prefix + '.u16'):
            return cr2, None, 'nooutput'
        if (meta.get('iso') or 99999) > MAX_ISO:
            return cr2, None, f'ISO {meta.get("iso")}'
        w, h = meta['width'], meta['height']
        a = np.fromfile(prefix + '.u16', dtype='<u2').reshape(3, h, w)
        rnd = random.Random(os.path.basename(cr2))
        out, tries = [], 0
        while len(out) < CROPS and tries < CROPS * 10:
            tries += 1
            y, x = rnd.randrange(4, h - S - 4), rnd.randrange(4, w - S - 4)
            c = a[:, y:y + S, x:x + S]
            if (c == 65535).any(0).mean() > 0.05 or c.mean() < 0.004 * 65534:
                continue
            out.append(c.copy())
        return cr2, (meta, np.stack(out) if out else None), None
    except Exception as e:
        return cr2, None, 'error: ' + str(e)[:80]
    finally:
        for ext in ('.u16', '.json'):
            try:
                os.remove(prefix + ext)
            except OSError:
                pass


def md5_ok(z, k):
    import hashlib, urllib.request
    d = json.load(urllib.request.urlopen('https://zenodo.org/api/records/4153431', timeout=60))
    want = [f['checksum'] for f in d['files'] if f['key'] == f'source_CR2_{k}.zip'][0].split(':')[1]
    h = hashlib.md5()
    with open(z, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest() == want


def salvage(z, ex):
    """Read members from local headers, keep only those whose CRC checks out (a damaged download loses only
    the members it hits)."""
    import struct, zlib
    good = bad = 0
    with open(z, 'rb') as f:
        data_len = os.path.getsize(z)
        pos = 0
        while pos < data_len - 30:
            f.seek(pos)
            hd = f.read(30)
            if hd[:4] != b'PK':
                # find the next local header
                f.seek(pos + 1)
                buf = f.read(1 << 24)
                i = buf.find(b'PK')
                if i < 0:
                    pos += 1 + len(buf) - 3
                    if not buf:
                        break
                    continue
                pos += 1 + i
                continue
            _, ver, flag, meth, _, _, crc, csz, usz, nl, xl = struct.unpack('<IHHHHHIIIHH', hd)
            name = f.read(nl).decode('utf8', 'replace')
            extra = f.read(xl)
            j = 0
            while j + 4 <= len(extra):           # zip64 sizes
                hid, hl = struct.unpack('<HH', extra[j:j + 4])
                if hid == 1:
                    vals = list(struct.unpack('<' + 'Q' * (hl // 8), extra[j + 4:j + 4 + (hl // 8) * 8]))
                    if usz == 0xFFFFFFFF and vals:
                        usz = vals.pop(0)
                    if csz == 0xFFFFFFFF and vals:
                        csz = vals.pop(0)
                j += 4 + hl
            start = pos + 30 + nl + xl
            if flag & 8 or csz > 200_000_000 or start + csz > data_len:
                pos += 4; bad += 1; continue
            raw = f.read(csz)
            try:
                out = zlib.decompress(raw, -15) if meth == 8 else raw if meth == 0 else None
            except zlib.error:
                out = None
            if out is not None and len(out) == usz and zlib.crc32(out) == crc:
                if name.lower().endswith('.cr2'):
                    open(os.path.join(ex, os.path.basename(name)), 'wb').write(out)
                    good += 1
                pos = start + csz
            else:
                bad += 1; pos += 4
    return good, bad


def one_zip(k, src='zenodo'):
    done_flag = f'{OUT}/done_{k}'
    if os.path.exists(done_flag):
        print(f'zip {k}: already done', flush=True); return
    z = f'{OUT}/source_CR2_{k}.zip'
    print(f'zip {k}: downloading from {src}', flush=True)
    if not (os.path.exists(z) and os.path.getsize(z) == sizes()[f'source_CR2_{k}.zip']) and not download(k, src, z):
        print(f'zip {k}: download failed', flush=True); return
    ex = f'{OUT}/x_{k}'
    os.makedirs(ex, exist_ok=True)
    if md5_ok(z, k):
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith('.cr2')]
            print(f'zip {k}: {len(names)} CR2 files, checksum ok', flush=True)
            for n in names:
                dst = os.path.join(ex, os.path.basename(n))
                with zf.open(n) as zsrc, open(dst, 'wb') as d:
                    shutil.copyfileobj(zsrc, d, 1 << 20)
    else:
        good, bad = salvage(z, ex)
        print(f'zip {k}: checksum mismatch (damaged download), salvaged {good} intact CR2 files, {bad} damaged spots', flush=True)
    files = sorted({os.path.normcase(p): p for p in glob.glob(f'{ex}/*.CR2') + glob.glob(f'{ex}/*.cr2')}.values())
    kept, rows, skipped = [], [], {}
    with ThreadPoolExecutor(6) as pool:
        for cr2, res, why in pool.map(lambda f: crops_of(f, ex), files):
            if why:
                key = why.split(':')[0].split(' ')[0]
                skipped[key] = skipped.get(key, 0) + 1
            elif res and res[1] is not None:
                meta, c = res
                kept.append(c)
                rows.append([os.path.basename(cr2), meta.get('make'), meta.get('model'), meta.get('iso'), len(c),
                             'CC-BY-4.0', 'https://zenodo.org/records/4153431'])
            try:
                os.remove(cr2)
            except OSError:
                pass
    if kept:
        np.save(f'{OUT}/crops_{k}.npy', np.concatenate(kept))
    new = not os.path.exists(f'{OUT}/sources.csv')
    with open(f'{OUT}/sources.csv', 'a', newline='', encoding='utf8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['file', 'make', 'model', 'iso', 'crops', 'licence', 'url'])
        w.writerows(rows)
    shutil.rmtree(ex, ignore_errors=True)
    os.remove(z)
    open(done_flag, 'w').write('ok')
    print(f'zip {k}: kept {len(rows)} pictures ({sum(len(c) for c in kept)} crops), skipped {skipped}', flush=True)


if __name__ == '__main__':
    # two workers, one per server, take zips from a shared queue; each processes the zip it downloaded
    import queue, threading
    os.makedirs(OUT, exist_ok=True)
    q = queue.Queue()
    for k in range(10):
        q.put(k)
    lock = threading.Lock()
    def worker(src):
        while True:
            try:
                k = q.get_nowait()
            except queue.Empty:
                return
            try:
                one_zip(k, src)
            except Exception as e:
                print(f'zip {k}: error {e}', flush=True)
    ts = [threading.Thread(target=worker, args=(s,)) for s in SOURCES]
    for t in ts: t.start()
    for t in ts: t.join()
    print('all zips done', flush=True)
