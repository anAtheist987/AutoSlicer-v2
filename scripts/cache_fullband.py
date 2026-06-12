"""Cache log-mel for FULL-BAND audio (top-level 歌回【主播】/*.m4s).

Output: data/cache/mel_fb/<dir>/<stem>.npy  (same 80-mel/50fps format as 8k caches).
Only processes streamers we actually train on, plus the gold-labeled Azusa VODs.

Usage: python scripts/cache_fullband.py --gpu 1
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/root/Autoslicer')
sys.path.insert(0, str(ROOT / 'src'))

DIRS = ['歌回【东雪莲】', '歌回【小柔channel】', '歌回【月隐空夜】', '歌回【猫雷】',
        '歌回【阿梓从小就很可爱】']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}'
    from features import FeatureConfig, extract_logmel_file
    fcfg = FeatureConfig()

    jobs = []
    for d in DIRS:
        for m4s in sorted((ROOT / 'data/raw' / d).glob('*.m4s')):
            jobs.append((d, m4s))
    print(f'{len(jobs)} files', flush=True)
    for d, m4s in jobs:
        out = ROOT / 'data/cache/mel_fb' / d / (m4s.stem + '.npy')
        if out.exists():
            continue
        t0 = time.time()
        try:
            mel = extract_logmel_file(m4s, fcfg, device)
        except Exception as e:
            print(f'SKIP {m4s.name}: {e}', flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, mel)
        print(f'{d}/{m4s.stem}: [{mel.shape[1]}f] {time.time()-t0:.0f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
