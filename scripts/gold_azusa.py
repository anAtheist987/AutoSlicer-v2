"""Gold-label pipeline for 阿梓 (human-annotated Audition marker CSVs).

1. Parse 阿梓 标注/*.csv (tab-separated, Start/Duration as M:SS.mmm or H:MM:SS.mmm).
2. Cache log-mel for the labeled wavs.
3. Build gold manifests: 4 train + 2 val 阿梓 VODs, plus other-streamer VODs as
   zero-label hard negatives (focus = their pseudo-singing segments).

Usage: python scripts/gold_azusa.py --gpu 0
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/Autoslicer')
sys.path.insert(0, str(ROOT / 'src'))

ANN = ROOT / 'data/raw/阿梓 标注'
OUT_FPS = 3.125
VAL_VODS = {'1188184012-1-30280', '1190452750-1-30280'}
EXCLUDE = {'1237149092-1-30280'}  # empty csv: annotation status unknown


def read_time(t: str) -> float:
    parts = t.strip().split(':')
    parts = [float(p) for p in parts]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def parse_marks(p: Path) -> list[tuple[float, float]]:
    segs = []
    with open(p, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            s = read_time(row['Start'])
            d = read_time(row['Duration'])
            if d > 0:
                segs.append((s, s + d))
    return sorted(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}'

    from features import FeatureConfig, extract_logmel_file
    fcfg = FeatureConfig()

    mel_dir = ROOT / 'data/cache/mel/阿梓'
    lab_dir = ROOT / 'data/cache/labels_gold/阿梓'
    mel_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for csv_p in sorted(ANN.glob('*.csv')):
        stem = csv_p.stem
        if stem in EXCLUDE:
            print(f'skip {stem} (excluded)')
            continue
        wav_p = ANN / 'temp' / f'{stem}.wav'
        if not wav_p.exists():
            print(f'skip {stem} (no wav)')
            continue
        segs = parse_marks(csv_p)
        mel_p = mel_dir / f'{stem}.npy'
        if not mel_p.exists():
            mel = extract_logmel_file(wav_p, fcfg, device)
            np.save(mel_p, mel)
        else:
            mel = np.load(mel_p, mmap_mode='r')
        n_out = mel.shape[1] // 16
        lab = np.zeros(n_out, dtype=np.float32)
        for s, e in segs:
            lab[int(s * OUT_FPS): int(e * OUT_FPS)] = 1.0
        lab_p = lab_dir / f'{stem}.lab.npy'
        np.save(lab_p, lab)
        items.append({'mel': str(mel_p), 'label': str(lab_p), 'vod': f'阿梓/{stem}',
                      'n_songs': len(segs), 'pos_frac': round(float(lab.mean()), 3)})
        print(f'{stem}: {len(segs)} songs, pos={lab.mean():.1%}, {n_out/OUT_FPS/3600:.1f}h')

    train = [it for it in items if Path(it['mel']).stem not in VAL_VODS]
    val = [it for it in items if Path(it['mel']).stem in VAL_VODS]

    # other-streamer hard negatives (reuse existing zero labels + focus)
    old_train = json.loads((ROOT / 'data/processed/manifest_train.json').read_text())
    others = [it for it in old_train
              if 'focus' in it and not it['vod'].startswith('synth')
              and '东雪莲' not in it['vod']]
    # 东雪莲 now also counts as "another singer" for the 阿梓 task: add her vods as
    # zero-label negatives with her pseudo-singing as focus
    zero_dir = ROOT / 'data/cache/labels_zero_for_azusa'
    for mel_p in sorted((ROOT / 'data/cache/mel/歌回【东雪莲】').glob('*.npy')):
        stem = mel_p.stem
        lab_src = ROOT / 'data/cache/labels/歌回【东雪莲】' / f'{stem}.lab.npy'
        segs_p = ROOT / 'data/cache/labels/歌回【东雪莲】' / f'{stem}.segs.json'
        if not lab_src.exists():
            continue
        z_p = zero_dir / f'{stem}.lab0.npy'
        z_p.parent.mkdir(parents=True, exist_ok=True)
        if not z_p.exists():
            np.save(z_p, np.zeros(len(np.load(lab_src)), dtype=np.float32))
        focus = [[s['start'], s['end']] for s in json.loads(segs_p.read_text()) if s['positive']] \
            if segs_p.exists() else None
        others.append({'mel': str(mel_p), 'label': str(z_p), 'vod': f'歌回【东雪莲】/{stem}',
                       'focus': focus})

    train_all = train + others
    out = ROOT / 'data/processed'
    (out / 'gold_train.json').write_text(json.dumps(train_all, ensure_ascii=False, indent=1))
    (out / 'gold_val.json').write_text(json.dumps(val, ensure_ascii=False, indent=1))
    print(f'gold_train: {len(train_all)} items ({len(train)} gold + {len(others)} negatives)')
    print(f'gold_val: {len(val)} items: {[it["vod"] for it in val]}')


if __name__ == '__main__':
    main()
