"""Build a noise profile for a camera folder (calib_dark + calib_flat).

Per ISO: read noise and dark offset per CFA channel (pair differences of dark frames), row banding
(temporal, from dark frames), and gain in e-/DN (pair differences of flats at the same exposure, centre
crop, per brightness level; level means/% of range reported so linearity can be judged).

Usage: python calibrate.py <camera_dir> <profile_name>"""
import json, os, sys
from collections import defaultdict
import numpy as np
from common import load_packed, exif

S = 300  # half-size of the centre crop, in cells


def crop(x):
    _, H, W = x.shape
    return x[:, H // 2 - S:H // 2 + S, W // 2 - S:W // 2 + S]


def main(cam, name):
    darks, flats = defaultdict(list), defaultdict(list)
    for sub, dct in (('calib_dark', darks), ('calib_flat', flats)):
        for f in sorted(os.listdir(f'{cam}/{sub}')):
            m = exif(f'{cam}/{sub}/{f}')
            key = m['iso'] if sub == 'calib_dark' else (m['iso'], m['exposure'])
            dct[key].append(f'{cam}/{sub}/{f}')
    prof = {'camera': None, 'units': 'DN above black; variance = signal/gain + read_noise^2', 'iso': {}}
    print('ISO   | read DN (R G1 G2 B)        | row DN | flat level  %range  gain e-/DN (pairs)')
    for iso in sorted(darks):
        fr = [load_packed(p) for p in darks[iso][:8]]
        rng = fr[0][1]
        D = [crop(x) for x, _ in fr]
        rn = [float((D[0][c] - D[1][c]).std() / np.sqrt(2)) for c in range(4)]
        off = [float(np.mean([d[c].mean() for d in D])) for c in range(4)]
        mean = np.mean(D, axis=0)
        rows = []
        for d in D[:4]:
            e = d - mean
            rm = e.mean(2)
            rows.append(np.sqrt(max(rm.var() - e.var() / e.shape[2], 0)))
        row = float(np.mean(rows) * np.sqrt(len(D) / (len(D) - 1)))
        gains, levels = [], []
        for (fi, t), paths in sorted(flats.items()):
            if fi != iso or len(paths) < 2:
                continue
            a, b = crop(load_packed(paths[0])[0]), crop(load_packed(paths[1])[0])
            if max(a.max(), b.max()) >= rng * 0.98:
                levels.append((t, 'clipped')); continue
            g = []
            for c in (1, 2):
                m = 0.5 * (a[c] + b[c]).mean() - off[c]
                v = (a[c] - b[c]).var() / 2
                if v > rn[c] ** 2 and m > 20 * rn[c]:
                    g.append(m / (v - rn[c] ** 2))
            mean_g = 0.5 * (a[1] + a[2]).mean() - off[1]
            if g:
                gains.append(float(np.mean(g)))
                levels.append((t, f'{100 * mean_g / rng:5.1f}%  {np.mean(g):.4f}'))
            else:
                levels.append((t, f'{100 * mean_g / rng:5.1f}%  too dark'))
        prof['iso'][str(iso)] = {
            'white_range': rng, 'read_noise_dn': [round(x, 3) for x in rn], 'dark_offset_dn': [round(x, 3) for x in off],
            'row_noise_dn': round(row, 4), 'gain_e_per_dn': round(float(np.median(gains)), 5) if gains else None}
        lv = '; '.join(f'{("1/%d" % round(1 / t)) if t < 1 else t}s {s}' for t, s in levels) or 'no flats'
        print(f'{iso:>5} | {" ".join(f"{x:5.2f}" for x in rn)} | {row:6.3f} | {lv}', flush=True)
    prof['camera'] = exif(darks[min(darks)][0])['model']
    json.dump(prof, open(f'{cam}/{name}.json', 'w'), indent=1)
    print('saved', f'{cam}/{name}.json')


if __name__ == '__main__':
    main(*sys.argv[1:])
