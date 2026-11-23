"""Export a train_cf.py checkpoint as a Chromaforge AI Denoise model folder:
model.onnx + denoise-model.json, then run the app's own installation self-test (rawdump selftest).

Usage: python export_cf.py <run> [--version 0.1.0] [--ckpt best.pt]"""
import argparse, hashlib, json, os, subprocess, sys
import torch

import model as M  # noqa: E402

from common import WORK as HERE, ROOT, CODE  # noqa: E402  data/work folder (see common.py)
RAWDUMP = f'{HERE}/rawdump-target/release/lc-denoise-rawdump.exe'
TILE, OVERLAP = 512, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--ckpt', default='best.pt')
    ap.add_argument('--version', default='0.1.0')
    a = ap.parse_args()
    s = torch.load(f'{HERE}/runs/{a.run}/{a.ckpt}', map_location='cpu')
    net = M.BayerUNet(tuple(s['widths'])); net.load_state_dict(s['ema'])
    out = f'{HERE}/export/chromaclean-bayer-{a.version}'
    os.makedirs(out, exist_ok=True)
    onnx_path = f'{out}/model.onnx'
    M.export_onnx(net, onnx_path, TILE)
    data = open(onnx_path, 'rb').read()
    meta = json.load(open(f'{HERE}/clean_crops.json'))
    n_sony = sum(p.lower().endswith('.arw') for p in meta['images'])
    n_oly = len(meta['images']) - n_sony
    mix = json.load(open(f'{HERE}/clean_rgb.json')) if os.path.exists(f'{HERE}/clean_rgb.json') and 'clean_rgb' in open(f'{HERE}/runs/{a.run}/train.log').read() else None
    if mix:
        c = mix['counts']
        data_txt = (f"{mix['count']} quality-checked low-ISO crops: the author's own Sony A7 IV and Olympus E-M10 IV raws "
                    f"({c.get('sony', 0) + c.get('olympus', 0)} crops), CC0 raws from raw.pixls.us ({c.get('rpu', 0)} crops, many "
                    f"cameras; contributors credited in credits/raw_pixls_us_cc0.csv of the source repository)")
        if c.get('cube'):
            data_txt += (f", and {c['cube']} crops from the Cube++ dataset (CC BY 4.0: E. Ershov et al., \"The Cube++ "
                         "Illumination Estimation Dataset\", IEEE Access 2020, https://zenodo.org/records/4153431)")
    else:
        data_txt = (f"the author's own photos ({n_sony} Sony A7 IV and {n_oly} Olympus E-M10 IV low-ISO raws, "
                    f"{meta['count']} crops)")
    manifest = {
        'id': 'chromaclean-bayer',
        'name': 'ChromaClean Denoise (Bayer)',
        'version': a.version,
        'licence': {
            'name': 'MIT OR Apache-2.0',
            'commercial': 'yes',
            'url': 'https://opensource.org/license/mit',
            'notice': ("ChromaClean denoise model. Weights licensed MIT OR Apache-2.0. Trained on " + data_txt +
                       ", with synthetic sensor noise calibrated on a Sony A7 IV and Olympus E-M10 IV (dark frames and "
                       "flat fields, numbers only)."),
        },
        'source': 'https://github.com/AudiFormer/ChromaClean-Denoise',
        'sha256': hashlib.sha256(data).hexdigest(),
        'sizeBytes': len(data),
        'provenance': (f"Trained with a {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M-weight U-Net (Chromaforge "
                       f"tools/denoise-train model.py, MIT) for {s['step']} steps on " + data_txt + "; Poisson-Gaussian noise with row banding, calibrated on a Sony A7 IV; validated on "
                       "real A7 IV and E-M10 IV ISO 200-25600 frames not used in training."),
        'domain': 'bayerToRgb',
        'tile': TILE,
        'overlap': OVERLAP,
        'gain': {'kind': 'none'},
    }
    man_path = f'{out}/denoise-model.json'
    json.dump(manifest, open(man_path, 'w', encoding='utf8'), indent=2)
    print(json.dumps({k: manifest[k] for k in ('id', 'version', 'sha256', 'sizeBytes')}))
    p = subprocess.run([RAWDUMP, 'selftest', onnx_path, man_path], capture_output=True, text=True)
    print(p.stdout.strip(), p.stderr.strip())
    print('folder:', out)
    return p.returncode


if __name__ == '__main__':
    sys.exit(main())
