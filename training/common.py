"""Shared helpers: RAW loading into packed RGGB planes, calibration profile."""
import json
import numpy as np
import rawpy
import exifread

import os as _os
# data lives outside the repository: $CHROMACLEAN_DATA, or the folder that holds the repository
ROOT = _os.environ.get('CHROMACLEAN_DATA') or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
WORK = f'{ROOT}/train'          # crops, real pairs, runs, exports, the rawdump helper build
CODE = _os.path.dirname(_os.path.abspath(__file__))


def exif(path):
    with open(path, 'rb') as fh:
        t = exifread.process_file(fh, details=False)
    g = lambda k: str(t.get(k, '')).strip()
    def sec(s):
        if '/' in s:
            a, b = s.split('/')
            return float(a) / float(b)
        return float(s) if s else 0.0
    return {'model': g('Image Model'), 'iso': int(g('EXIF ISOSpeedRatings') or 0),
            'exposure': sec(g('EXIF ExposureTime'))}


def load_packed(path):
    """Return (planes[4,H/2,W/2] float32 in DN above black, white_range).

    Planes are always ordered R, G1, G2, B regardless of the sensor's CFA phase.
    Values are NOT clipped at zero (real raw noise goes below black)."""
    with rawpy.imread(path) as r:
        if r.raw_pattern is None:
            raise ValueError('not a Bayer raw')
        a = r.raw_image_visible.astype(np.float32)
        pat = r.raw_pattern
        desc = r.color_desc.decode()  # e.g. 'RGBG'
        bl = np.array(r.black_level_per_channel, np.float32)
        wl = float(r.white_level)
        if r.raw_type != rawpy.RawType.Flat or r.raw_pattern is None:
            raise ValueError('not a Bayer raw')
    # map pattern index -> colour letter
    letters = [desc[pat[y, x]] for y in range(2) for x in range(2)]
    pos = {}
    for l, yx in zip(letters, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        if l == 'G':
            l = 'G1' if 'G1' not in pos else 'G2'
        pos[l] = yx
    H, W = a.shape
    H, W = H // 2 * 2, W // 2 * 2
    out = []
    for key in ('R', 'G1', 'G2', 'B'):
        y, x = pos[key]
        plane = a[y:H:2, x:W:2]
        ci = pat[y, x]
        out.append(plane - bl[ci])
    blk = float(np.mean(bl))
    return np.stack(out), wl - blk


def load_profile(path=f'{ROOT}/sony/profile_sony_a7iv.json'):
    p = json.load(open(path))
    return {int(k): v for k, v in p['iso'].items()}


def noise_params(profile, iso, white_range):
    """Noise in normalised units (signal / white_range):
    var = a * x + b, with a = 1/(gain*range), b = (read/range)^2 per channel."""
    e = profile[iso]
    a = 1.0 / (e['gain_e_per_dn'] * white_range)
    b = (np.array(e['read_noise_dn'], np.float32) / white_range) ** 2
    return a, b
