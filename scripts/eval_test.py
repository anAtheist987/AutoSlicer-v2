"""One-shot test-set evaluation with FIXED post-processing parameters.

The parameters must come from tuning on the validation set only; this script
just applies them to dumped test probabilities and reports metrics. Run it
once per protocol — repeated runs with different params would turn the test
set into a second validation set.

Usage:
  python scripts/eval_test.py --probs-dir runs/v2_final/test_probs \
      [--on 0.6 --off 0.3 --median 5 --gap 30 --min-song 60]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from metrics import event_metrics  # noqa: E402
from postprocess import PostProcessConfig, mask_to_runs, probs_to_segments  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs-dir', required=True)
    ap.add_argument('--on', type=float, default=0.6)
    ap.add_argument('--off', type=float, default=0.3)
    ap.add_argument('--median', type=float, default=3.0)
    ap.add_argument('--gap', type=float, default=45.0)
    ap.add_argument('--min-song', type=float, default=50.0)
    args = ap.parse_args()

    probs_dir = Path(args.probs_dir)
    meta = json.loads((probs_dir / 'dump_meta.json').read_text())
    fps = meta['out_fps']
    pcfg = PostProcessConfig(frame_rate=fps, on_threshold=args.on, off_threshold=args.off,
                             median_s=args.median, max_gap_s=args.gap,
                             min_song_s=args.min_song)
    frame = {'tp': 0, 'fp': 0, 'fn': 0}
    ev_pred, ev_true, t_off = [], [], 0.0
    per_vod = []
    for it in meta['items']:
        probs = np.load(probs_dir / f"{it['stem']}.probs.npy")
        lab = np.load(it['label'])
        n = min(len(probs), len(lab))
        probs, lab = probs[:n], lab[:n]
        p, t = probs >= 0.5, lab >= 0.5
        frame['tp'] += int((p & t).sum())
        frame['fp'] += int((p & ~t).sum())
        frame['fn'] += int((~p & t).sum())
        segs = probs_to_segments(probs, pcfg)
        pred = [(s['start'] + t_off, s['end'] + t_off) for s in segs]
        true = []
        for a, b in mask_to_runs(t):
            s, e = a / fps, b / fps
            if true and s - true[-1][1] <= pcfg.merge_gap_s:
                true[-1] = (true[-1][0], e)
            else:
                true.append((s, e))
        m = event_metrics([(s - t_off, e - t_off) for s, e in pred],
                          true, iou_threshold=0.5)
        per_vod.append({'vod': it['vod'], 'n_true': len(true), 'n_pred': len(pred),
                        'event_f1': round(m['event_f1'], 4)})
        ev_pred += pred
        ev_true += [(s + t_off, e + t_off) for s, e in true]
        t_off += n / fps + 3600
    tp, fp, fn = frame['tp'], frame['fp'], frame['fn']
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    out = {'params': vars(args) | {'probs_dir': str(probs_dir)},
           'frame_precision': round(prec, 4), 'frame_recall': round(rec, 4),
           'frame_f1': round(2 * prec * rec / max(1e-9, prec + rec), 4),
           **{k: round(v, 4) if isinstance(v, float) else v
              for k, v in event_metrics(ev_pred, ev_true, iou_threshold=0.5).items()},
           'per_vod': per_vod}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (probs_dir / 'test_result.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
