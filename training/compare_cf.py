"""Compare Chromaforge denoisers on real scene frames (one per ISO, a centred 512-cell tile).

Contenders: the built-in block-matching denoiser (via cf-bench), installed ONNX models (upstream LightCraft,
RawNIND) and our exported model. Score: tone-mapped PSNR after binning each RGB output to cells, against the
aligned base-ISO reference binned the same way. Also writes side-by-side crops.

Usage: python compare_cf.py [ours_model_dir] [scene]"""
import json, os, subprocess, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

from common import WORK as HERE, ROOT, CODE  # noqa: E402  data/work folder (see common.py)
BENCH = f'{ROOT}/cf-bench/target/release/cf-bench.exe'
MODELS = os.path.join(os.environ['APPDATA'], 'Chromaforge', 'denoise-models')
T = 512


def tone(x):
    return np.sign(x) * np.log1p(100 * np.abs(x)) / np.log1p(100)


def psnr(x, y, m=None):
    e = (tone(x) - tone(y)) ** 2
    if m is not None:
        return float(-10 * np.log10((e * m).sum() / (m.sum() * e.shape[0])))
    return float(-10 * np.log10(e.mean()))


def still_mask(n, r):
    """cells where the scene did not move between the noisy frame and the reference: the low-passed difference
    stays within 5 robust deviations of its own spread (movement shows as smooth, structured difference)"""
    from scipy.ndimage import uniform_filter, maximum_filter
    lp = uniform_filter((n[1] + n[2]) - (r[1] + r[2]), 15)
    mad = 1.4826 * np.median(np.abs(lp - np.median(lp)))
    moved = maximum_filter(np.abs(lp) > 5 * mad, 9)
    return (~moved).astype(np.float32)


def to_rgb(p):
    return np.stack([p[0], 0.5 * (p[1] + p[2]), p[3]])


def bin2(rgb):
    c, h, w = rgb.shape
    return rgb.reshape(c, h // 2, 2, w // 2, 2).mean((2, 4))


def run_onnx(path, gain, tile):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    def f(planes):
        out = sess.run(None, {name: planes[None].astype(np.float32)})[0][0]
        if gain.get('kind') == 'matchMean':          # as the app does: per-channel scale measured, clamped to nominal
            nom, dev = gain['nominal'], gain['max_deviation']
            cin = to_rgb(planes)
            for c in range(3):
                g = bin2(out)[c].mean() / max(cin[c].mean(), 1e-6)
                out[c] /= np.clip(g, nom * (1 - dev), nom * (1 + dev))
        return out
    return f


def run_builtin(planes):
    tmp = os.path.join(HERE, '_bench_in.f32'); tmo = os.path.join(HERE, '_bench_out.f32')
    planes.astype('<f4').tofile(tmp)
    p = subprocess.run([BENCH, tmp, tmo, str(planes.shape[1])], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    out = np.fromfile(tmo, '<f4').reshape(3, 2 * planes.shape[1], 2 * planes.shape[1])
    os.remove(tmp); os.remove(tmo)
    return out


def render(rgb, wb, expo):
    x = np.clip(np.stack([rgb[0] * wb[0], rgb[1], rgb[2] * wb[2]], -1) * expo, 0, 1) ** (1 / 2.2)
    return (x * 255 + 0.5).astype(np.uint8)


def main(ours=None, scene='scene_001'):
    d = f'{HERE}/pairs/{scene}'
    idx = json.load(open(f'{d}/index.json'))
    contenders = {'builtin': run_builtin}
    for mid in sorted(os.listdir(MODELS)):
        mf = os.path.join(MODELS, mid, 'denoise-model.json')
        if os.path.exists(mf):
            m = json.load(open(mf))
            contenders[mid] = run_onnx(os.path.join(MODELS, mid, 'model.onnx'), m['gain'], m['tile'])
    for o in (ours.split(',') if ours else []):
        m = json.load(open(os.path.join(o, 'denoise-model.json')))
        contenders[f'ours {m["version"]}'] = run_onnx(os.path.join(o, 'model.onnx'), m['gain'], m['tile'])
    out_dir = f'{HERE}/compare_{scene}'; os.makedirs(out_dir, exist_ok=True)
    base = np.load(f'{d}/ref_base.npy').astype(np.float32)
    g = (base[1].mean() + base[2].mean()) / 2
    wb = (g / base[0].mean(), 1.0, g / base[3].mean())
    expo = 0.18 / max(np.median((base[1] + base[2]) / 2), 1e-4)
    g1 = base[1]
    gy = np.abs(np.diff(g1, axis=0))[:, :-1] + np.abs(np.diff(g1, axis=1))[:-1, :]
    best = -1
    for yy in range(0, g1.shape[0] - T, 128):
        for xx in range(0, g1.shape[1] - T, 128):
            v = gy[yy:yy + T, xx:xx + T].mean()
            if v > best: best, detail_at = v, (yy, xx)
    if os.environ.get('TILE_AT') == 'detail':
        out_dir += '_detail'; os.makedirs(out_dir, exist_ok=True)
    table, seen = {}, set()
    print('ISO    still noisy  ' + '  '.join(f'{k[:18]:>18}' for k in contenders))
    for fr in idx['frames']:
        iso = fr['iso']
        if iso in seen:
            continue
        seen.add(iso)
        n = np.load(f'{d}/noisy_{fr["file"][:-4]}.npy').astype(np.float32)
        r = np.load(f'{d}/ref_{fr["file"][:-4]}.npy').astype(np.float32)
        _, H, W = n.shape
        y0, x0 = (H - T) // 2, (W - T) // 2
        if os.environ.get('TILE_AT') == 'detail':
            y0, x0 = detail_at
        mask = still_mask(n, r)[y0:y0 + T, x0:x0 + T]
        n, r = n[:, y0:y0 + T, x0:x0 + T], r[:, y0:y0 + T, x0:x0 + T]
        ref = to_rgb(r)
        row = {'noisy': psnr(np.minimum(to_rgb(n), 1), ref, mask), 'still': float(mask.mean())}
        imgs = [np.repeat(np.repeat(to_rgb(n), 2, 1), 2, 2)]
        for k, f in contenders.items():
            o = f(n)
            row[k] = psnr(bin2(o), ref, mask)
            imgs.append(o)
        imgs.append(np.repeat(np.repeat(ref, 2, 1), 2, 2))
        table[iso] = row
        print(f'{iso:>5}  {row["still"]:4.2f}  {row["noisy"]:5.2f}  ' + '  '.join(f'{row[k]:18.2f}' for k in contenders), flush=True)
        # a 384-pixel crop from the middle of the tile, every contender side by side
        c0 = T - 192
        Image.fromarray(np.concatenate([render(im[:, c0:c0 + 384, c0:c0 + 384], wb, expo) for im in imgs], 1)).save(
            f'{out_dir}/iso{iso:05d}.png')
    json.dump({'order': ['noisy'] + list(contenders) + ['reference'], 'psnr': table}, open(f'{out_dir}/results.json', 'w'), indent=1)
    print('panels: noisy | ' + ' | '.join(contenders) + ' | reference ->', out_dir)


if __name__ == '__main__':
    main(*sys.argv[1:])
