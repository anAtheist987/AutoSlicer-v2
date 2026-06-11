"""Batch HTDemucs 2-stem separation of all VOD wavs.

Stems are stored mono/16kHz/PCM16 (all downstream use is 16k mono):
  data/cache/stems/<streamer>/<stem>.vocals.wav
  data/cache/stems/<streamer>/<stem>.accomp.wav

Usage: python scripts/separate_all.py --gpu 2 --shard 0 --num-shards 3
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

ROOT = Path('/root/Autoslicer')
sys.path.insert(0, str(ROOT / 'src'))
from features import load_audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    args = ap.parse_args()
    device = f'cuda:{args.gpu}'

    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    model = get_model('htdemucs').to(device).eval()
    vocals_idx = model.sources.index('vocals')

    wavs = sorted((ROOT / 'data/raw/data_8k').rglob('*.wav'))
    wavs = [w for i, w in enumerate(wavs) if i % args.num_shards == args.shard]
    print(f'shard {args.shard}: {len(wavs)} files', flush=True)

    for w in wavs:
        streamer = w.parent.name
        out_dir = ROOT / 'data/cache/stems' / streamer
        voc_p = out_dir / (w.stem + '.vocals.wav')
        acc_p = out_dir / (w.stem + '.accomp.wav')
        if voc_p.exists() and acc_p.exists():
            continue
        t0 = time.time()
        wav = load_audio(w, 44100)  # mono
        st = torch.stack([wav, wav])  # demucs wants stereo [2, L]
        ref = st.mean(0).std() + 1e-8
        st = (st - st.mean()) / ref
        with torch.no_grad():
            # chunk long files: apply_model has its own segmenting; pass whole tensor
            sources = apply_model(model, st[None], device=device, split=True,
                                  overlap=0.1, progress=False)[0]
        sources = sources * ref + st.mean()
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, path in [(vocals_idx, voc_p)]:
            stem = sources[idx].mean(0)
            stem16 = torchaudio.functional.resample(stem, 44100, 16000)
            sf.write(path, stem16.clamp(-1, 1).numpy(), 16000, subtype='PCM_16')
        # accompaniment = sum of all non-vocal sources
        acc = sources[[i for i in range(len(model.sources)) if i != vocals_idx]].sum(0).mean(0)
        acc16 = torchaudio.functional.resample(acc, 44100, 16000)
        sf.write(acc_p, acc16.clamp(-1, 1).numpy(), 16000, subtype='PCM_16')
        dur = len(wav) / 44100
        print(f'{streamer}/{w.stem}: {dur/60:.0f}min in {time.time()-t0:.0f}s '
              f'({dur/(time.time()-t0):.0f}x RT)', flush=True)
    print('SHARD DONE', flush=True)


if __name__ == '__main__':
    main()
