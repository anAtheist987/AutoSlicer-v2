"""Build multiclass labels + v2 manifests (full-band + 8k dual-domain).

Class encoding (float arrays): -1 ignore | 0 none/speech | 1 A(阿梓) singing |
2 other-person singing (incl. played recordings) | 3 music w/o vocals | 4 not-A-unknown.

Sources:
  gold 阿梓 VODs : human songs -> 1; outside songs, from separated-stem features:
                   acc&!voc -> 3, voc&!acc -> 0, none -> 0, voc&acc -> -1
                   (could be unlabeled singing; safer to ignore than mislabel)
  other streamers: their pseudo-sing segments -> 2 (edges -1); acc-only -> 3; else 0
  synthetic      : remix_other -> 2, accomp_only -> 3, voc_mismatch(东雪莲 voice) -> 2

Manifests:
  data/processed/v2_train.json : gold 4 VODs x {8k, fb} + others x {8k, fb} + synth
  data/processed/v2_val.json   : gold val 2 VODs, FULL-BAND mel, BINARY labels (1/0)

Usage: python scripts/build_v2_labels.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/Autoslicer')
sys.path.insert(0, str(ROOT / 'src'))

OUT_FPS = 3.125
VOC_ON_DB, ACC_ON_DB = -38.0, -45.0
VAL_VODS = {'1188184012-1-30280', '1190452750-1-30280'}
EXCLUDE = {'1237149092-1-30280'}
AZ_FB_DIR = ROOT / 'data/raw/歌回【阿梓从小就很可爱】'
OTHERS = ['歌回【东雪莲】', '歌回【小柔channel】', '歌回【月隐空夜】', '歌回【猫雷】']

sys.path.insert(0, str(ROOT / 'scripts'))
from gold_azusa import parse_marks  # noqa: E402

OUT = ROOT / 'data/cache/labels_v2'


def feat_classes(feat: np.ndarray) -> np.ndarray:
    """[T,4](voc_db,acc_db,rstd,sust) -> base class array {0,3} + voc&acc mask separately."""
    voc = feat[:, 0].astype(np.float32) > VOC_ON_DB
    acc = feat[:, 1].astype(np.float32) > ACC_ON_DB
    cls = np.zeros(len(feat), dtype=np.float32)        # default 0 none/speech
    cls[acc & ~voc] = 3.0
    return cls, (voc & acc)


def gold_class_label(stem: str, n_out: int) -> np.ndarray:
    segs = parse_marks(ROOT / 'data/raw/阿梓 标注' / f'{stem}.csv')
    feat_p = ROOT / 'data/cache/labels/阿梓' / f'{stem}.feat.npy'
    if feat_p.exists():
        feat = np.load(feat_p)
        cls, both = feat_classes(feat)
        n = min(n_out, len(cls))
        lab = np.full(n_out, 0.0, dtype=np.float32)
        lab[:n] = cls[:n]
        lab[:n][both[:n]] = -1.0   # unlabeled vocal-over-music: ignore
    else:
        lab = np.full(n_out, 4.0, dtype=np.float32)  # not-A-unknown fallback
    for s, e in segs:
        a, b = int(s * OUT_FPS), int(e * OUT_FPS)
        lab[max(0, a): min(n_out, b)] = 1.0
    return lab


def other_class_label(streamer: str, stem: str, n_out: int) -> np.ndarray | None:
    lab_dir = ROOT / 'data/cache/labels' / streamer
    feat_p = lab_dir / f'{stem}.feat.npy'
    segs_p = lab_dir / f'{stem}.segs.json'
    if not feat_p.exists():
        return None
    cls, both = feat_classes(np.load(feat_p))
    n = min(n_out, len(cls))
    lab = np.zeros(n_out, dtype=np.float32)
    lab[:n] = cls[:n]
    lab[:n][both[:n]] = 4.0   # vocal over music but didn't pass song filters: not-A anyway
    buf = int(8 * OUT_FPS)
    if segs_p.exists():
        for s in json.loads(segs_p.read_text()):
            if not s['positive']:
                continue
            a, b = int(s['start'] * OUT_FPS), int(s['end'] * OUT_FPS)
            lab[max(0, a - buf): min(n_out, b + buf)] = -1.0
            lab[max(0, a + buf): min(n_out, max(0, b - buf))] = 2.0
    return lab


def main():
    train, val = [], []
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- gold 阿梓, both domains
    for csv_p in sorted((ROOT / 'data/raw/阿梓 标注').glob('*.csv')):
        stem = csv_p.stem
        if stem in EXCLUDE:
            continue
        domains = [('8k', ROOT / 'data/cache/mel/阿梓' / f'{stem}.npy')]
        fb = ROOT / 'data/cache/mel_fb/歌回【阿梓从小就很可爱】' / f'{stem}.npy'
        if fb.exists():
            domains.append(('fb', fb))
        for dom, mel_p in domains:
            if not mel_p.exists():
                continue
            n_out = np.load(mel_p, mmap_mode='r').shape[1] // 16
            if stem in VAL_VODS:
                if dom == 'fb':  # val: full-band, binary labels
                    lab = np.zeros(n_out, dtype=np.float32)
                    for s, e in parse_marks(csv_p):
                        lab[int(s * OUT_FPS): int(e * OUT_FPS)] = 1.0
                    lp = OUT / f'az_{stem}.val.npy'
                    np.save(lp, lab)
                    val.append({'mel': str(mel_p), 'label': str(lp), 'vod': f'阿梓fb/{stem}'})
                continue
            lab = gold_class_label(stem, n_out)
            lp = OUT / f'az_{stem}.{dom}.npy'
            np.save(lp, lab)
            train.append({'mel': str(mel_p), 'label': str(lp), 'vod': f'阿梓{dom}/{stem}'})

    # ---- other streamers, both domains
    for st in OTHERS:
        for dom, mdir in [('8k', ROOT / 'data/cache/mel' / st),
                          ('fb', ROOT / 'data/cache/mel_fb' / st)]:
            for mel_p in sorted(mdir.glob('*.npy')) if mdir.exists() else []:
                stem = mel_p.stem
                n_out = np.load(mel_p, mmap_mode='r').shape[1] // 16
                lab = other_class_label(st, stem, n_out)
                if lab is None:
                    continue
                lp = OUT / f'{st}_{stem}.{dom}.npy'
                np.save(lp, lab)
                segs_p = ROOT / 'data/cache/labels' / st / f'{stem}.segs.json'
                focus = [[s['start'], s['end']] for s in json.loads(segs_p.read_text())
                         if s['positive']] if segs_p.exists() else None
                train.append({'mel': str(mel_p), 'label': str(lp),
                              'vod': f'{st}{dom}/{stem}', 'focus': focus or None})

    # ---- synthetic remap: 1(voc_mismatch=东雪莲 sings)->2, others keep 0->? see header
    for mel_p in sorted((ROOT / 'data/cache/synth').glob('*.mel.npy')):
        name = mel_p.name.replace('.mel.npy', '')
        n_out = np.load(mel_p, mmap_mode='r').shape[1] // 16
        if name.startswith('voc_mismatch') or name.startswith('remix_other'):
            v = 2.0
        elif name.startswith('accomp_only'):
            v = 3.0
        else:
            continue
        lp = OUT / f'synth_{name}.npy'
        np.save(lp, np.full(n_out, v, dtype=np.float32))
        train.append({'mel': str(mel_p), 'label': str(lp), 'vod': f'synthv2/{name}'})

    out = ROOT / 'data/processed'
    (out / 'v2_train.json').write_text(json.dumps(train, ensure_ascii=False, indent=1))
    (out / 'v2_val.json').write_text(json.dumps(val, ensure_ascii=False, indent=1))
    from collections import Counter
    c = Counter(it['vod'].split('/')[0] for it in train)
    print(f'v2_train: {len(train)} items {dict(c)}')
    print(f'v2_val: {len(val)} items {[it["vod"] for it in val]}')


if __name__ == '__main__':
    main()
