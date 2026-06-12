"""Grid-search post-processing constants on dumped val probabilities.

Optimizes event-F1 (IoU>=0.5, ground truth gets the same merge semantics),
reports frame-F1 at the chosen operating point too.

Usage: python scripts/tune_postprocess.py --probs-dir runs/crnn_a/val_probs
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from metrics import event_metrics  # noqa: E402
from postprocess import PostProcessConfig, mask_to_runs, probs_to_segments  # noqa: E402


def merged_truth(lab: np.ndarray, fps: float, merge_gap_s: float):
    runs = []
    for a, b in mask_to_runs((lab > 0.5)):
        s, e = a / fps, b / fps
        if runs and s - runs[-1][1] <= merge_gap_s:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    return [tuple(r) for r in runs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs-dir', required=True)
    ap.add_argument('--iou', type=float, default=0.5)
    args = ap.parse_args()
    d = Path(args.probs_dir)
    meta = json.loads((d / 'dump_meta.json').read_text())
    fps = meta['out_fps']

    vods = []
    for it in meta['items']:
        probs = np.load(d / f"{it['stem']}.probs.npy")
        lab = np.load(it['label'])
        n = min(len(probs), len(lab))
        vods.append((it['vod'], probs[:n], lab[:n]))

    grid = {
        'on_threshold': [0.5, 0.6, 0.7],
        'off_threshold': [0.3, 0.4],
        'median_s': [3.0, 6.0],
        'max_gap_s': [30.0, 45.0, 60.0],
        'min_song_s': [40.0, 60.0],
    }
    best = None
    results = []
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid.keys(), vals))
        cfg = PostProcessConfig(frame_rate=fps, **kw)
        ev_pred, ev_true = [], []
        off = 0.0
        for vod, probs, lab in vods:
            segs = probs_to_segments(probs, cfg)
            ev_pred += [(s['start'] + off, s['end'] + off) for s in segs]
            ev_true += [(a + off, b + off) for a, b in merged_truth(lab, fps, cfg.merge_gap_s)]
            off += len(lab) / fps + 3600
        m = event_metrics(ev_pred, ev_true, iou_threshold=args.iou)
        rec = {**kw, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
        results.append(rec)
        if best is None or m['event_f1'] > best[0]:
            best = (m['event_f1'], rec)
    results.sort(key=lambda r: -r['event_f1'])
    (d / 'tune_results.json').write_text(json.dumps(results[:20], indent=1))
    print('BEST:', json.dumps(best[1], indent=1))
    # per-vod report at best config
    kw = {k: best[1][k] for k in grid}
    cfg = PostProcessConfig(frame_rate=fps, **kw)
    for vod, probs, lab in vods:
        segs = probs_to_segments(probs, cfg)
        truth = merged_truth(lab, fps, cfg.merge_gap_s)
        m = event_metrics([(s['start'], s['end']) for s in segs], truth, iou_threshold=args.iou)
        valid = lab >= 0
        p = (probs >= 0.5)
        t = (lab >= 0.5)
        tp = (p & t & valid).sum(); fp = (p & ~t & valid).sum(); fn = (~p & t & valid).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        print(f"{vod}: events P={m['event_precision']:.2f} R={m['event_recall']:.2f} "
              f"F1={m['event_f1']:.2f} ({m['event_tp']}/{len(truth)}) frameF1={f1:.3f}")


if __name__ == '__main__':
    main()
