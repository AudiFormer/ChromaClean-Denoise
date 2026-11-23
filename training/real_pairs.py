"""Turn a scene folder into aligned, brightness-matched real pairs.

For each noisy frame: the base-ISO reference (mean of base frames) is shifted
sub-pixel onto the frame and scaled per channel to its brightness.
Output: pairs/<scene>/ref_<file>.npy (aligned, matched reference) + noisy_<file>.npy,
both float16 packed RGGB normalised to white range, and index.json."""
import json, os, sys
import numpy as np
from scipy.ndimage import fourier_shift
from common import load_packed, exif, ROOT

from common import WORK as HERE, ROOT, CODE  # noqa: E402  data/work folder (see common.py)


def lowpass(x, k=4):
    H, W = x.shape[0] // k * k, x.shape[1] // k * k
    return x[:H, :W].reshape(H // k, k, W // k, k).mean((1, 3))


def phase_shift(a, b):
    """shift (dy,dx) such that b ~= a shifted by (dy,dx)"""
    A = np.fft.fft2(a - a.mean()); B = np.fft.fft2(b - b.mean())
    R = B * np.conj(A); R /= np.abs(R) + 1e-9
    c = np.fft.ifft2(R).real
    H, W = c.shape
    y, x = np.unravel_index(c.argmax(), c.shape)
    def sub(m, i, n):
        l, cc, r = m[(i - 1) % n], m[i], m[(i + 1) % n]
        d = l - 2 * cc + r
        return 0.0 if d == 0 else 0.5 * (l - r) / d
    dy = y + sub(c[:, x], y, H); dx = x + sub(c[y, :], x, W)
    if dy > H / 2: dy -= H
    if dx > W / 2: dx -= W
    return dy, dx


def shift_planes(x, dy, dx):
    return np.stack([np.fft.ifftn(fourier_shift(np.fft.fftn(p), (dy, dx))).real for p in x]).astype(np.float32)


def main(scene, cam='sony'):
    d = f'{ROOT}/{cam}/scenes/{scene}'
    out = f'{HERE}/pairs/{scene}' if cam == 'sony' else f'{HERE}/pairs/{cam}_{scene}'; os.makedirs(out, exist_ok=True)
    files = sorted(os.listdir(d))
    meta = {f: exif(f'{d}/{f}') for f in files}
    base_iso = min(m['iso'] for m in meta.values())
    base = [f for f in files if meta[f]['iso'] == base_iso]
    frames = {}
    rng = None
    for f in files:
        x, rng = load_packed(f'{d}/{f}')
        frames[f] = x / rng
    ref = np.mean([frames[f] for f in base], axis=0)
    _, H, W = ref.shape
    cy, cx, s = H // 2, W // 2, 512
    crop = lambda p: p[cy - s:cy + s, cx - s:cx + s]
    g_ref = lowpass(crop(ref[1] + ref[2]))
    index = {'scene': scene, 'camera': cam, 'white_range': rng, 'base_iso': base_iso, 'base_frames': base, 'frames': []}
    for f in files:
        if f in base:
            continue
        x = frames[f]
        dy, dx = phase_shift(g_ref, lowpass(crop(x[1] + x[2])))
        dy, dx = dy * 4, dx * 4                      # low-pass factor -> packed-plane pixels
        cand = []
        for sgn in (1, -1):
            r = shift_planes(ref, sgn * dy, sgn * dx)
            err = np.mean((lowpass(crop(x[1]), 2) - lowpass(crop(r[1]), 2))[8:-8, 8:-8] ** 2)
            cand.append((err, sgn, r))
        cand.sort(key=lambda c: c[0])
        _, sgn, r = cand[0]
        # per-channel brightness match on unclipped, non-dark pixels
        scales = []
        for c in range(4):
            m = (r[c] > 0.02) & (r[c] < 0.9) & (x[c] < 0.95)
            scales.append(float((x[c][m] * r[c][m]).sum() / (r[c][m] ** 2).sum()))
            r[c] *= scales[-1]
        # drop the wrap-around border the Fourier shift creates
        b = 8
        np.save(f'{out}/ref_{f[:-4]}.npy', r[:, b:-b, b:-b].astype(np.float16))
        np.save(f'{out}/noisy_{f[:-4]}.npy', x[:, b:-b, b:-b].astype(np.float16))
        index['frames'].append({'file': f, 'iso': meta[f]['iso'], 'shift_packed': [sgn * dy, sgn * dx], 'scales': scales})
        print(f, meta[f]['iso'], f'shift {sgn*dy:+.3f},{sgn*dx:+.3f} (packed px)', 'scales', [round(v, 4) for v in scales], flush=True)
    np.save(f'{out}/ref_base.npy', ref[:, 8:-8, 8:-8].astype(np.float16))
    json.dump(index, open(f'{out}/index.json', 'w'), indent=1)


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['scene_001']))
