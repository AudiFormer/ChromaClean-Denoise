"""Train a Chromaforge-compatible AI Denoise model (contract `bayerToRgb`, gain `none`).

Network: upstream's exportable BayerUNet (tools/denoise-train/model.py), blind (no noise-level input):
RGGB planes [1,4,t,t] (black subtracted, white 1, not white balanced) -> camera RGB [1,3,2t,2t].

Target: the clean raw binned per 2x2 cell into one RGB pixel (R, mean G, B), so no demosaic is needed for the
reference and its noise halves again. The binned picture is re-mosaicked and gets calibrated sensor noise:
Poisson shot noise, Gaussian read noise (b = read_e^2 * a^2), row banding (fraction of read noise, measured
on the A7 IV), DN quantisation, white clipping. White-balance jitter and exposure darkening widen the range.

Validation: the real scene_001 frames (never trained on), compared after binning the output back to cells.

Usage: python train_cf.py [--steps N] [--run NAME]"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

import model as M  # noqa: E402  upstream's exportable network

from common import WORK as HERE, ROOT, CODE  # noqa: E402  data/work folder (see common.py)
dev = 'cuda'
WIDTHS = (32, 64, 128, 256, 512)


def tone(x):
    return torch.sign(x) * torch.log1p(100 * x.abs()) / math.log1p(100)


def to_rgb(p):          # packed (n,4,h,w) -> binned RGB (n,3,h,w)
    return torch.stack([p[:, 0], 0.5 * (p[:, 1] + p[:, 2]), p[:, 3]], 1)


def mosaic(rgb):        # (n,3,2t,2t) -> (n,4,t,t)
    return torch.stack([rgb[:, 0, 0::2, 0::2], rgb[:, 1, 0::2, 1::2], rgb[:, 1, 1::2, 0::2], rgb[:, 2, 1::2, 1::2]], 1)


def bin2(rgb):          # (n,3,2h,2w) -> (n,3,h,w)
    return F.avg_pool2d(rgb, 2)


def u(lo, hi, n, gen):
    return torch.exp(torch.empty(n, device=dev).uniform_(math.log(lo), math.log(hi), generator=gen))


def make_batch(data, n, side, gen, weights=None):
    """side = RGB size (= 2t). data: packed RGGB (N,4,S,S) or binned RGB (N,3,S,S).
    Returns noisy planes, RGB target, valid mask, exposure k."""
    N, C, S, _ = data.shape
    if weights is None:
        idx = torch.randint(0, N, (n,), device=dev, generator=gen)
    else:
        idx = torch.multinomial(weights, n, replacement=True, generator=gen)
    y = torch.randint(0, S - side + 1, (n,), device=dev, generator=gen)
    x = torch.randint(0, S - side + 1, (n,), device=dev, generator=gen)
    ar = torch.arange(side, device=dev)
    yy = (y[:, None] + ar)[:, None, :, None]; xx = (x[:, None] + ar)[:, None, None, :]
    if data.device.type == 'cpu':          # large sets stay in system RAM: gather on the CPU, send the batch
        i4, c4, y4, x4 = idx.cpu()[:, None, None, None], torch.arange(C)[None, :, None, None], yy.cpu(), xx.cpu()
        p = data[i4, c4, y4, x4].to(dev, non_blocking=True).float()
    else:
        p = data[idx[:, None, None, None], torch.arange(C, device=dev)[None, :, None, None], yy, xx].float()
    valid = (p.max(1).values < 0.97).float()
    rgb = to_rgb(p) if C == 4 else p
    # rotations / flips (free in RGB: the mosaic is made afterwards)
    k = int(torch.randint(0, 4, (1,), generator=gen, device=dev))
    rgb, valid = torch.rot90(rgb, k, (2, 3)), torch.rot90(valid, k, (1, 2))
    if torch.rand(1, generator=gen, device=dev) < 0.5:
        rgb, valid = rgb.flip(3), valid.flip(2)
    # white-balance jitter: other cameras / lights put R and B at other levels
    gains = torch.ones(n, 3, 1, 1, device=dev)
    gains[:, 0, 0, 0] = torch.empty(n, device=dev).uniform_(0.6, 1.4, generator=gen)
    gains[:, 2, 0, 0] = torch.empty(n, device=dev).uniform_(0.6, 1.4, generator=gen)
    rgb = rgb * gains
    expo = u(1 / 16, 1.2, n, gen)
    target = rgb * expo[:, None, None, None]
    valid = valid * (target.max(1).values < 0.98).float()
    target = target.clamp(max=1.0)
    clean = mosaic(target)
    # noise: a = shot coefficient (ISO ~50 .. ~100k), b = (read electrons * a)^2
    a = u(1e-5, 2.5e-2, n, gen)
    rn_e = u(0.8, 6.0, n, gen)
    rn = (rn_e * a)[:, None] * torch.empty(n, 4, device=dev).uniform_(0.92, 1.04, generator=gen)
    a4 = a[:, None, None, None]
    noisy = torch.poisson(clean.clamp(min=0) / a4, generator=gen) * a4
    noisy = noisy + torch.randn(clean.shape, device=dev, generator=gen) * rn[:, :, None, None]
    ratio = u(0.02, 0.12, n, gen)
    H = clean.shape[2]
    rows = 0.7 * torch.randn(n, 1, H, 1, device=dev, generator=gen) + 0.7 * torch.randn(n, 4, H, 1, device=dev, generator=gen)
    noisy = noisy + rows * (ratio[:, None] * rn)[:, :, None, None]
    R = u(4000, 16000, n, gen)[:, None, None, None]          # sensor range in DN: quantisation step 1/R
    noisy = (torch.round(noisy * R) / R).clamp(max=1.0)
    return noisy, target, valid, expo


def loss_fn(out, target, valid, expo):
    kk = expo[:, None, None, None]
    o, t = out / kk, target / kk
    m = valid[:, None]
    d = m.sum().clamp(min=1) * 3
    return ((tone(o) - tone(t)).abs() * m).sum() / d + 0.25 * ((o - t).abs() * m).sum() / d


# ---------------- validation on real pairs ----------------
def load_val(scenes=('scene_001', 'olympus_scene_001'), size=768):
    out = []
    for sc in scenes:
        if os.path.exists(f'{HERE}/pairs/{sc}/index.json'):
            out += [(f'{sc[:3]}{iso}', n, r) for iso, n, r in load_val_one(sc, size)]
    return out


def load_val_one(scene='scene_001', size=768):
    d = f'{HERE}/pairs/{scene}'
    idx = json.load(open(f'{d}/index.json'))
    out, seen = [], set()
    for fr in idx['frames']:
        if fr['iso'] in seen:
            continue
        seen.add(fr['iso'])
        n = np.load(f'{d}/noisy_{fr["file"][:-4]}.npy'); r = np.load(f'{d}/ref_{fr["file"][:-4]}.npy')
        _, H, W = n.shape
        y0, x0 = (H - size) // 2, (W - size) // 2
        out.append((fr['iso'], torch.tensor(n[:, y0:y0 + size, x0:x0 + size]).float()[None].to(dev),
                    torch.tensor(r[:, y0:y0 + size, x0:x0 + size]).float()[None].to(dev)))
    return out


def psnr(x, y):
    return (-10 * torch.log10(F.mse_loss(tone(x), tone(y)))).item()


@torch.no_grad()
def validate(net, val):
    net.eval(); res = {}
    for iso, n, r in val:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = net(n).float()
        res[iso] = (psnr(to_rgb(n).clamp(max=1), to_rgb(r)), psnr(bin2(p), to_rgb(r)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--side', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--run', default='cf1')
    ap.add_argument('--eval_every', type=int, default=5000)
    ap.add_argument('--data', default='clean_crops', help='clean_crops (packed) or clean_rgb (merged, weighted)')
    args = ap.parse_args()
    run = f'{HERE}/runs/{args.run}'; os.makedirs(run, exist_ok=True)
    log = open(f'{run}/train.log', 'a')
    def say(s):
        print(s, flush=True); log.write(s + '\n'); log.flush()
    meta = json.load(open(f'{HERE}/{args.data}.json'))
    data = np.load(f'{HERE}/{args.data}.npy', mmap_mode='r')[:meta['count']]
    say(f'loading {meta["count"]} clean crops ({args.data}) to GPU...')
    data = torch.from_numpy(np.array(data))
    if data.numel() * 2 < 6e9:
        data = data.to(dev)
    say(f'clean data on {data.device} ({data.numel() * 2 / 1e9:.1f} GB)')
    weights = torch.tensor(meta['weight'], device=dev, dtype=torch.float32) if 'weight' in meta else None
    val = load_val()
    torch.manual_seed(0)
    gen = torch.Generator(device=dev); gen.manual_seed(1)
    net = M.BayerUNet(WIDTHS).to(dev)
    ema = M.BayerUNet(WIDTHS).to(dev); ema.load_state_dict(net.state_dict()); ema.requires_grad_(False)
    say(f'params {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M, args {vars(args)}')
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    best, t0, run_loss, nl = -1e9, time.time(), 0.0, 0
    warm = 2000
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g['lr'] = lr
        noisy, target, valid, expo = make_batch(data, args.batch, args.side, gen, weights)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(noisy)
        loss = loss_fn(out.float(), target, valid, expo)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(d).add_(pn.detach(), alpha=1 - d)
        if step % 50 == 0:
            run_loss += loss.item(); nl += 1
        if step % 1000 == 0:
            say(f'step {step} loss {run_loss / max(nl, 1):.5f} lr {lr:.2e} {(time.time() - t0) / step * 1000:.0f} ms/step')
            run_loss, nl = 0.0, 0
        if step % args.eval_every == 0 or step == args.steps:
            res = validate(ema, val)
            gain = np.mean([d - n for n, d in res.values()])
            say(f'  VAL step {step}: ' + ' '.join(f'{iso}:{n:.1f}->{d:.1f}' for iso, (n, d) in res.items()) + f'  mean gain {gain:+.2f} dB')
            ck = {'ema': ema.state_dict(), 'step': step, 'widths': WIDTHS, 'val': res}
            torch.save(ck, f'{run}/last.pt')
            if gain > best:
                best = gain; torch.save(ck, f'{run}/best.pt')
    say(f'done, best mean gain {best:+.2f} dB')


if __name__ == '__main__':
    main()
