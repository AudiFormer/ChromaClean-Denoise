"""Before/after examples of the released model on public-domain (CC0) high-ISO raws from raw.pixls.us.

Runs the exported model.onnx itself (onnxruntime, as Chromaforge feeds it: RGGB planes, black subtracted, white 1,
no white balance) on one 512-cell tile per picture, at its most detailed spot. Both sides get the same colour
treatment: the camera's as-shot white balance, its colour matrix to sRGB, one exposure, sRGB tone curve.
The noisy side is the raw shown with a plain bilinear demosaic.

Usage: python make_examples.py <model_dir> <raws_dir> <out_dir>"""
import json, os, sys
import numpy as np
import onnxruntime as ort
import rawpy
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom
from common import load_packed

T = 512
XYZ_SRGB = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])


def colour(path):
    with rawpy.imread(path) as r:
        wb = np.array(r.camera_whitebalance[:3], np.float64)
        cam_xyz = np.array(r.rgb_xyz_matrix[:3], np.float64)
        iso = None
    wb = wb / wb[1] if wb[1] > 0 else np.ones(3)
    if not np.isfinite(wb).all() or wb.min() <= 0:
        wb = np.ones(3)
    if np.abs(cam_xyz).sum() < 1e-6:
        m = np.eye(3)
    else:
        cam_rgb = cam_xyz @ XYZ_SRGB
        cam_rgb /= cam_rgb.sum(1, keepdims=True)
        m = np.linalg.inv(cam_rgb)
    return wb, m


def show(rgb, wb, m, expo):
    x = np.minimum(rgb * wb[:, None, None], 1.0)   # clipped raw highlights stay neutral after white balance
    x = np.einsum('ij,jhw->ihw', m, x) * expo
    x = np.clip(x, 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return (np.transpose(x, (1, 2, 0)) * 255 + 0.5).astype(np.uint8)


def bilinear(planes):
    """plain demosaic of R, G1, G2, B planes to a full-size RGB picture (for the 'before' side)"""
    r = zoom(planes[0], 2, order=1)
    g = zoom(0.5 * (planes[1] + planes[2]), 2, order=1)
    b = zoom(planes[3], 2, order=1)
    return np.stack([r, g, b])


def main(model_dir, raws, out):
    os.makedirs(out, exist_ok=True)
    man = json.load(open(f'{model_dir}/denoise-model.json'))
    sess = ort.InferenceSession(f'{model_dir}/model.onnx', providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    src = json.load(open(f'{raws}/../sources.json'))
    try:
        font = ImageFont.truetype('arialbd.ttf', 26); small = ImageFont.truetype('arial.ttf', 20)
    except OSError:
        font = small = ImageFont.load_default()
    made = []
    for s in src:
        path = f'{raws}/{s["file"]}'
        try:
            planes, rng = load_packed(path)
        except ValueError as e:
            print(s['file'], 'skipped:', e); continue
        planes = planes / rng
        _, H, W = planes.shape
        # most detailed 512-cell spot: mid-scale detail of a smoothed green (smoothing removes the noise)
        g = gaussian_filter(0.5 * (planes[1] + planes[2]), 2)
        det = np.abs(gaussian_filter(g, 1) - gaussian_filter(g, 6))
        lum = gaussian_filter(g, 30)
        score = gaussian_filter(det, 80) * (lum > 0.02) * (lum < 0.6)
        b = T // 2 + 8
        score[:b], score[-b:], score[:, :b], score[:, -b:] = 0, 0, 0, 0
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
        y0, x0 = cy - T // 2, cx - T // 2
        tile = planes[:, y0:y0 + T, x0:x0 + T].astype(np.float32)
        den = sess.run(None, {name: tile[None]})[0][0]
        noisy = bilinear(tile)
        wb, m = colour(path)
        lum_d = np.einsum('ij,jhw->ihw', m, den * wb[:, None, None])[1]
        expo = 0.8 / max(float(np.percentile(lum_d, 99)), 1e-4)
        c = 128                                    # keep the middle 768 px: away from the tile edge
        a = show(noisy, wb, m, expo)[c:-c, c:-c]; d = show(den, wb, m, expo)[c:-c, c:-c]
        h, w, _ = a.shape
        img = Image.new('RGB', (2 * w + 12, h + 64), (22, 22, 22))
        img.paste(Image.fromarray(a), (0, 64)); img.paste(Image.fromarray(d), (w + 12, 64))
        dr = ImageDraw.Draw(img)
        dr.text((10, 6), f'{s["make"]} {s["model"]}  -  ISO {s["iso"]}  -  100% crop', fill=(255, 255, 255), font=font)
        dr.text((10, 38), 'Raw (bilinear demosaic)', fill=(200, 200, 200), font=small)
        dr.text((w + 22, 38), f'{man["name"]} {man["version"]}', fill=(200, 200, 200), font=small)
        fn = f'{s["make"]}_{s["model"]}_ISO{s["iso"]}.jpg'.replace(' ', '_')
        img.save(f'{out}/{fn}', quality=92)
        made.append(dict(s, example=fn, crop_cells=[int(y0), int(x0), T]))
        print(fn, flush=True)
    json.dump(made, open(f'{out}/examples.json', 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:4])
