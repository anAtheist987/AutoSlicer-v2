"""Build train/val manifests from cached mel + pseudo labels (+ synthetic clips).

- Target streamer VODs: labels from pseudo_label.py (1/0/-1). Two held out for val.
- Other streamers: label = all zeros ("not A"); their own pseudo-singing segments are
  attached as `focus` regions so crops oversample the hard "someone else sings" parts.
  One other-streamer VOD goes to val to measure false-positive resistance.
- Synthetic clips from make_synthetic.py are appended to train.

Usage: python scripts/build_manifests.py --target 歌回【东雪莲】
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/root/Autoslicer')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='歌回【东雪莲】')
    ap.add_argument('--val-vods', nargs='*', default=None,
                    help='target-vod stems for validation; default: 2 with most positives')
    ap.add_argument('--out-dir', default='data/processed')
    ap.add_argument('--feature', default='mel', choices=['mel', 'emb'])
    args = ap.parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fps = 3.125 if args.feature == 'mel' else 1.5625
    suffix = '' if args.feature == 'mel' else '_emb'

    mel_dir = ROOT / 'data/cache' / args.feature
    lab_dir = ROOT / ('data/cache/labels' if args.feature == 'mel' else 'data/cache/emb_labels')
    streamers = sorted(d.name for d in mel_dir.iterdir() if d.is_dir() and d.name != 'synth')

    train, val = [], []

    # --- target streamer
    tgt_items = []
    for mel_p in sorted((mel_dir / args.target).glob('*.npy')):
        stem = mel_p.stem
        lab_name = f'{stem}.lab.npy' if args.feature == 'mel' else f'{stem}.npy'
        lab_p = lab_dir / args.target / lab_name
        if not lab_p.exists():
            print(f'no label for {stem}, skip')
            continue
        lab = np.load(lab_p)
        pos_s = float((lab > 0.5).sum() / out_fps)
        tgt_items.append({'mel': str(mel_p), 'label': str(lab_p), 'vod': f'{args.target}/{stem}',
                          'pos_seconds': round(pos_s, 1)})
    tgt_items.sort(key=lambda x: -x['pos_seconds'])
    if args.val_vods:
        val_set = set(args.val_vods)
        val_t = [it for it in tgt_items if Path(it['mel']).stem in val_set]
        train_t = [it for it in tgt_items if Path(it['mel']).stem not in val_set]
    else:
        # val: ranks 1 and 3 by positive amount (keep the richest for training)
        val_t = [it for i, it in enumerate(tgt_items) if i in (1, 3)]
        train_t = [it for i, it in enumerate(tgt_items) if i not in (1, 3)]
    train += train_t
    val += val_t

    # --- other streamers: all-zero labels + focus regions
    zero_dir = ROOT / ('data/cache/labels_zero' + suffix)
    segs_dir = ROOT / 'data/cache/labels'
    for st in streamers:
        if st == args.target:
            continue
        items = []
        for mel_p in sorted((mel_dir / st).glob('*.npy')):
            stem = mel_p.stem
            segs_p = segs_dir / st / f'{stem}.segs.json'
            lab_name = f'{stem}.lab.npy' if args.feature == 'mel' else f'{stem}.npy'
            lab_p = lab_dir / st / lab_name
            if not lab_p.exists():
                continue
            n = len(np.load(lab_p))
            z_p = zero_dir / st / f'{stem}.lab0.npy'
            z_p.parent.mkdir(parents=True, exist_ok=True)
            if not z_p.exists():
                np.save(z_p, np.zeros(n, dtype=np.float32))
            focus = []
            if segs_p.exists():
                focus = [[s['start'], s['end']] for s in json.loads(segs_p.read_text())
                         if s['positive']]
            items.append({'mel': str(mel_p), 'label': str(z_p), 'vod': f'{st}/{stem}',
                          'focus': focus or None})
        # longest VOD of the first other streamer -> val
        items.sort(key=lambda x: -Path(x['mel']).stat().st_size)
        train += items[1:]
        if items:
            val.append(items[0])

    # --- synthetic
    n_synth = 0
    if args.feature == 'mel':
        synth_p = ROOT / 'data/cache/synth/synth_manifest.json'
        if synth_p.exists():
            synth = json.loads(synth_p.read_text())
            train += synth
            n_synth = len(synth)
    else:
        for emb_p in sorted((ROOT / 'data/cache/emb/synth').glob('*.npy')):
            lab_p = ROOT / 'data/cache/emb_labels/synth' / emb_p.name
            if lab_p.exists():
                train.append({'mel': str(emb_p), 'label': str(lab_p), 'vod': f'synth/{emb_p.stem}'})
                n_synth += 1

    (out_dir / f'manifest_train{suffix}.json').write_text(json.dumps(train, ensure_ascii=False, indent=1))
    (out_dir / f'manifest_val{suffix}.json').write_text(json.dumps(val, ensure_ascii=False, indent=1))
    print(f'train: {len(train)} items ({n_synth} synthetic), val: {len(val)} items')
    for it in val:
        print('  val:', it['vod'])


if __name__ == '__main__':
    main()
