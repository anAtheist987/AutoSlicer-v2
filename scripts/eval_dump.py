"""Dump per-frame probabilities of a checkpoint over the val manifest.

Usage:
  python scripts/eval_dump.py --checkpoint runs/crnn_a/best.pt \
      --val-manifest data/processed/manifest_val.json --gpu 0 --out-dir runs/crnn_a/val_probs
Writes <out-dir>/<vodstem>.probs.npy + dump_meta.json (incl. wall-clock + RTF).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from models import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--val-manifest', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    targs = ckpt['args']
    model = build_model(targs['model'], **json.loads(targs.get('model_kwargs', '{}')))
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()
    input_type = targs.get('input_type', 'mel')
    out_fps = targs.get('out_fps') or (3.125 if input_type == 'mel' else 1.5625)
    ratio = 16 if input_type == 'mel' else 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(args.val_manifest).read_text())
    meta = {'checkpoint': args.checkpoint, 'out_fps': out_fps, 'items': []}
    chunk_out, overlap_out = 2048, 128
    for it in items:
        mel = np.load(it['mel'], mmap_mode='r')
        lab = np.load(it['label'])
        n_out = len(lab)
        probs = np.zeros(n_out, dtype=np.float32)
        weight = np.zeros(n_out, dtype=np.float32)
        t0 = time.time()
        step = chunk_out - overlap_out
        with torch.no_grad():
            for o in range(0, n_out, step):
                a, b = o, min(n_out, o + chunk_out)
                if input_type == 'mel':
                    x = np.asarray(mel[:, a * ratio: b * ratio], dtype=np.float32)
                else:
                    x = np.asarray(mel[a:b], dtype=np.float32)
                x = torch.from_numpy(x)[None].to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)[0].float().cpu().numpy()
                m = min(len(logits), b - a)
                if logits.ndim == 2:  # multiclass: P(class 1)
                    e = np.exp(logits[:m] - logits[:m].max(-1, keepdims=True))
                    p = e[:, 1] / e.sum(-1)
                else:
                    p = 1 / (1 + np.exp(-logits[:m]))
                probs[a: a + m] += p
                weight[a: a + m] += 1
                if b >= n_out:
                    break
        probs /= np.maximum(weight, 1)
        stem = Path(it['mel']).stem
        np.save(out_dir / f'{stem}.probs.npy', probs)
        dur = n_out / out_fps
        meta['items'].append({'vod': it['vod'], 'stem': stem, 'label': it['label'],
                              'duration_s': round(dur, 1),
                              'model_time_s': round(time.time() - t0, 2)})
        print(f"{it['vod']}: {dur/3600:.1f}h in {time.time()-t0:.1f}s", flush=True)
    (out_dir / 'dump_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
