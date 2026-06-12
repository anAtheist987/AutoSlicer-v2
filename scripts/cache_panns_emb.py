"""Cache PANNs CNN14_16k framewise embeddings (Path B inputs) + matching labels.

Embeddings: data/cache/emb/<streamer>/<stem>.npy  float16 [T', 2048] @1.5625 fps
Labels:     data/cache/emb_labels/<...>.npy       float32 [T'] from the 3.125 fps
            pseudo labels pooled 2:1 (any 1 -> 1, else any -1 -> -1, else 0).
Also processes data/cache/synth/*.wav.

Usage: python scripts/cache_panns_emb.py --gpu 1
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from features import load_audio  # noqa: E402
from panns import load_panns  # noqa: E402


@torch.no_grad()
def embed_file(path, model, device, chunk_s=300):
    sr = 16000
    wav = load_audio(path, sr)
    hop_emb = int(chunk_s * 1.5625)
    outs = []
    for s in range(0, len(wav), chunk_s * sr):
        seg = wav[s: s + chunk_s * sr]
        if len(seg) < sr:
            break
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model(seg[None].to(device))[0]
        outs.append(emb.float().cpu().to(torch.float16).numpy()[:hop_emb])
    return np.concatenate(outs, 0)


def pool_labels(lab: np.ndarray, n_emb: int) -> np.ndarray:
    n = min(len(lab) // 2, n_emb)
    pairs = lab[: n * 2].reshape(n, 2)
    out = np.zeros(n, dtype=np.float32)
    out[(pairs == 1).any(1)] = 1.0
    amb = (pairs == -1).any(1) & ~(pairs == 1).any(1)
    out[amb] = -1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}'
    model = load_panns(ROOT / 'pretrained/Cnn14_16k_mAP=0.438.pth', device)

    jobs = []  # (audio_path, emb_path, lab_src_path_or_None, emb_lab_path)
    for w in sorted((ROOT / 'data/raw/data_8k').rglob('*.wav')):
        st = w.parent.name
        lab_p = ROOT / 'data/cache/labels' / st / f'{w.stem}.lab.npy'
        if not lab_p.exists():
            continue
        jobs.append((w, ROOT / 'data/cache/emb' / st / f'{w.stem}.npy',
                     lab_p, ROOT / 'data/cache/emb_labels' / st / f'{w.stem}.npy'))
    for w in sorted((ROOT / 'data/cache/synth').glob('*.wav')):
        lab_p = w.with_name(w.stem + '.lab.npy')
        jobs.append((w, ROOT / 'data/cache/emb/synth' / f'{w.stem}.npy',
                     lab_p, ROOT / 'data/cache/emb_labels/synth' / f'{w.stem}.npy'))

    print(f'{len(jobs)} jobs', flush=True)
    for audio_p, emb_p, lab_src, lab_dst in jobs:
        if emb_p.exists() and lab_dst.exists():
            continue
        t0 = time.time()
        emb = embed_file(audio_p, model, device)
        emb_p.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_p, emb)
        lab = np.load(lab_src)
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(lab_dst, pool_labels(lab, len(emb)))
        print(f'{emb_p.parent.name}/{emb_p.stem}: {len(emb)} frames {time.time()-t0:.0f}s', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
