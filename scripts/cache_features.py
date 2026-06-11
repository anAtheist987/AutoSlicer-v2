"""Cache log-mel features + PANNs framewise tag probs for every VOD wav.

Outputs per VOD:
  data/cache/mel/<streamer>/<stem>.npy      float16 [80, T]    50 fps
  data/cache/panns/<streamer>/<stem>.npy    float16 [T', 527]  1.5625 fps (sigmoid probs)
Run:  python scripts/cache_features.py [--gpu 1] [--only-mel]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/Autoslicer')
sys.path.insert(0, str(ROOT / 'src'))
from features import FeatureConfig, LogMel, load_audio  # noqa: E402
from panns import load_panns  # noqa: E402


@torch.no_grad()
def panns_framewise_probs(wav16k: torch.Tensor, model, device, chunk_s=300) -> np.ndarray:
    import torch.nn.functional as F
    sr = 16000
    outs = []
    for s in range(0, len(wav16k), chunk_s * sr):
        seg = wav16k[s: s + chunk_s * sr]
        if len(seg) < sr:
            break
        with torch.autocast('cuda', dtype=torch.bfloat16):
            emb = model(seg[None].to(device))[0]          # [T', 2048] (pre-fc framewise)
            x = F.relu_(model.fc1(emb))
            probs = torch.sigmoid(model.fc_audioset(x))   # [T', 527]
        outs.append(probs.float().cpu().to(torch.float16).numpy())
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--only-mel', action='store_true')
    args = ap.parse_args()
    device = f'cuda:{args.gpu}'

    fcfg = FeatureConfig()
    frontend = LogMel(fcfg).to(device)
    panns = None if args.only_mel else load_panns(ROOT / 'pretrained/Cnn14_16k_mAP=0.438.pth', device)

    wavs = sorted((ROOT / 'data/raw/data_8k').rglob('*.wav'))
    print(f'{len(wavs)} wavs', flush=True)
    for w in wavs:
        streamer = w.parent.name
        mel_p = ROOT / 'data/cache/mel' / streamer / (w.stem + '.npy')
        pan_p = ROOT / 'data/cache/panns' / streamer / (w.stem + '.npy')
        if mel_p.exists() and (args.only_mel or pan_p.exists()):
            continue
        t0 = time.time()
        try:
            wav = load_audio(w, fcfg.sample_rate)
        except Exception as e:
            print(f'SKIP {w.name}: {e}', flush=True)
            continue
        if not mel_p.exists():
            mel_p.parent.mkdir(parents=True, exist_ok=True)
            hop = fcfg.hop_length
            chunk = (600 * fcfg.sample_rate // hop) * hop
            outs = []
            for s in range(0, len(wav), chunk):
                seg = wav[s: s + chunk].to(device)
                if len(seg) < fcfg.n_fft:
                    break
                outs.append(frontend(seg)[0].cpu().to(torch.float16).numpy())
            np.save(mel_p, np.concatenate(outs, 1))
        if panns is not None and not pan_p.exists():
            pan_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(pan_p, panns_framewise_probs(wav, panns, device))
        print(f'{streamer}/{w.stem}: {time.time()-t0:.0f}s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
