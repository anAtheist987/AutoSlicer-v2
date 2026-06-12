"""Pseudo-label generation from separated stems.

Per VOD, on the 3.125 fps output grid:
  features: vocal stem RMS(dB), accomp stem RMS(dB), YIN F0 rolling std (semitones)
  candidate singing frame = vocal active AND accomp active
  -> median smooth -> runs -> gap-fill(<=20s) -> min 45s -> segment filters:
       vocal density >= 0.4, sustained-note fraction >= 0.10, accomp level >= -48dB
  label = 1 inside passing segments, -1 (ignore) in +/-8s edge buffer, 0 elsewhere.

Outputs:
  data/cache/labels/<streamer>/<stem>.lab.npy    float32 [T_out] in {1,0,-1}
  data/cache/labels/<streamer>/<stem>.segs.json  pseudo segments (also for focus sampling)
  data/cache/labels/<streamer>/<stem>.feat.npy   float16 [T_out, 4] (voc_db, acc_db, roll_std, sustained)

Usage: python scripts/pseudo_label.py [--streamer 歌回【东雪莲】] [--workers 16]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

OUT_FPS = 3.125
HOP = 320          # 50 fps at 16k
POOL = 16          # 50/16 = 3.125

VOC_ON_DB = -38.0
ACC_ON_DB = -45.0
SEG_MIN_S = 45.0
GAP_FILL_S = 20.0
EDGE_IGNORE_S = 8.0
MIN_VOCAL_DENSITY = 0.40
MIN_SUSTAINED_FRAC = 0.10
MIN_ACC_DB = -48.0


def frame_db(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    r = librosa.feature.rms(y=x, frame_length=frame, hop_length=hop)[0]
    return 20 * np.log10(r + 1e-8)


def pool_mean(x: np.ndarray, k: int) -> np.ndarray:
    n = len(x) // k
    return x[: n * k].reshape(n, k).mean(1)


def process_vod(voc_path: str) -> str:
    voc_path = Path(voc_path)
    acc_path = voc_path.with_name(voc_path.name.replace('.vocals.wav', '.accomp.wav'))
    streamer = voc_path.parent.name
    stem = voc_path.name.replace('.vocals.wav', '')
    out_dir = ROOT / 'data/cache/labels' / streamer
    out_dir.mkdir(parents=True, exist_ok=True)
    lab_p = out_dir / f'{stem}.lab.npy'
    if lab_p.exists():
        return f'skip {streamer}/{stem}'

    voc, sr = sf.read(voc_path, dtype='float32')
    acc, _ = sf.read(acc_path, dtype='float32')

    voc_db50 = frame_db(voc, 1024, HOP)
    acc_db50 = frame_db(acc, 1024, HOP)

    # YIN over chunks (memory bound); 50 fps
    f0_parts = []
    chunk = 600 * sr
    ov = 1024
    for s in range(0, len(voc), chunk):
        seg = voc[max(0, s - ov): s + chunk]
        if len(seg) < 2048:
            break
        f0 = librosa.yin(seg, fmin=70, fmax=1000, sr=sr, frame_length=1024, hop_length=HOP)
        skip = (min(s, ov)) // HOP
        f0_parts.append(f0[skip:])
    f0 = np.concatenate(f0_parts)

    n50 = min(len(voc_db50), len(acc_db50), len(f0))
    voc_db50, acc_db50, f0 = voc_db50[:n50], acc_db50[:n50], f0[:n50]

    st = 12 * np.log2(np.maximum(f0, 1.0) / 55.0)
    k = 15  # 0.3 s
    pad = np.pad(st, (k // 2, k - 1 - k // 2), mode='edge')
    roll_std = np.lib.stride_tricks.sliding_window_view(pad, k).std(1)
    voiced50 = voc_db50 > VOC_ON_DB
    sustained50 = (voiced50 & (roll_std < 0.7)).astype(np.float32)

    # pool to 3.125 fps
    voc_db = pool_mean(voc_db50, POOL)
    acc_db = pool_mean(acc_db50, POOL)
    rstd = pool_mean(roll_std, POOL)
    sust = pool_mean(sustained50, POOL)
    n = len(voc_db)

    from scipy.ndimage import median_filter
    cand = (voc_db > VOC_ON_DB) & (acc_db > ACC_ON_DB)
    cand = median_filter(cand.astype(np.float32), size=int(3 * OUT_FPS) | 1) > 0.5

    # runs + gap fill + min duration
    d = np.diff(np.concatenate([[0], cand.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = []
    for a, b in zip(starts, ends):
        if runs and (a - runs[-1][1]) / OUT_FPS <= GAP_FILL_S:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    runs = [(a, b) for a, b in runs if (b - a) / OUT_FPS >= SEG_MIN_S]

    # segment-level filters
    segs = []
    for a, b in runs:
        vd = (voc_db[a:b] > VOC_ON_DB).mean()
        sf_ = sust[a:b][voc_db[a:b] > VOC_ON_DB]
        sfrac = float(sf_.mean()) if len(sf_) else 0.0
        am = float(np.median(acc_db[a:b]))
        ok = vd >= MIN_VOCAL_DENSITY and sfrac >= MIN_SUSTAINED_FRAC and am >= MIN_ACC_DB
        segs.append({'start': round(a / OUT_FPS, 2), 'end': round(b / OUT_FPS, 2),
                     'vocal_density': round(float(vd), 3), 'sustained_frac': round(sfrac, 3),
                     'acc_db': round(am, 1), 'positive': bool(ok)})

    lab = np.zeros(n, dtype=np.float32)
    buf = int(EDGE_IGNORE_S * OUT_FPS)
    for s_ in segs:
        if not s_['positive']:
            continue
        a, b = int(s_['start'] * OUT_FPS), int(s_['end'] * OUT_FPS)
        lab[max(0, a - buf): min(n, b + buf)] = -1.0
        lab[max(0, a + buf): max(0, b - buf)] = 1.0

    np.save(lab_p, lab)
    feat = np.stack([voc_db, acc_db, rstd, sust], 1).astype(np.float16)
    np.save(out_dir / f'{stem}.feat.npy', feat)
    (out_dir / f'{stem}.segs.json').write_text(json.dumps(segs, ensure_ascii=False, indent=1))
    pos = (lab > 0.5).mean()
    return f'{streamer}/{stem}: {n/OUT_FPS/3600:.1f}h pos={pos:.1%} segs={sum(s["positive"] for s in segs)}/{len(segs)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--streamer', default=None)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    stems_dir = ROOT / 'data/cache/stems'
    vocs = sorted(stems_dir.rglob('*.vocals.wav'))
    if args.streamer:
        vocs = [v for v in vocs if v.parent.name == args.streamer]
    print(f'{len(vocs)} vods', flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(process_vod, [str(v) for v in vocs]):
            print(r, flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
