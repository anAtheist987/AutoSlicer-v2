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
    ap.add_argument('--src-dir', default='data/raw/data_8k', help='directory to scan for *.wav')
    ap.add_argument('--out-name', default=None, help='stems subdir name; default = wav parent dir name')
    args = ap.parse_args()
    device = f'cuda:{args.gpu}'

    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    model = get_model('htdemucs').to(device).eval()
    vocals_idx = model.sources.index('vocals')

    wavs = sorted((ROOT / args.src_dir).rglob('*.wav'))
    wavs = [w for i, w in enumerate(wavs) if i % args.num_shards == args.shard]
    print(f'shard {args.shard}: {len(wavs)} files', flush=True)

    for w in wavs:
        streamer = args.out_name or w.parent.name
        out_dir = ROOT / 'data/cache/stems' / streamer
        voc_p = out_dir / (w.stem + '.vocals.wav')
        acc_p = out_dir / (w.stem + '.accomp.wav')
        if voc_p.exists() and acc_p.exists():
            continue
        t0 = time.time()
        wav = load_audio(w, 44100)  # mono
        # chunked processing with 1s crossfade keeps RAM bounded for multi-hour files
        # (whole-file sources tensor of a 2.8h VOD alone is ~14GB -> OOM with 3 procs)
        CH, XF = 480 * 44100, 44100
        out_dir.mkdir(parents=True, exist_ok=True)
        voc_parts, acc_parts = [], []
        prev_voc_tail = prev_acc_tail = None
        s = 0
        while s < len(wav):
            seg = wav[s: s + CH + XF]
            if len(seg) < 44100:
                break
            st = torch.stack([seg, seg])
            ref = st.mean(0).std() + 1e-8
            stn = (st - st.mean()) / ref
            with torch.no_grad():
                sources = apply_model(model, stn[None], device=device, split=True,
                                      overlap=0.1, progress=False)[0]
            sources = sources * ref + st.mean()
            voc = sources[vocals_idx].mean(0)
            acc = sources[[i for i in range(len(model.sources)) if i != vocals_idx]].sum(0).mean(0)
            if prev_voc_tail is not None:
                fade = torch.linspace(0, 1, min(XF, len(voc)))
                voc[:len(fade)] = voc[:len(fade)] * fade + prev_voc_tail[:len(fade)] * (1 - fade)
                acc[:len(fade)] = acc[:len(fade)] * fade + prev_acc_tail[:len(fade)] * (1 - fade)
            body = min(CH, len(voc))
            voc_parts.append(voc[:body])
            acc_parts.append(acc[:body])
            prev_voc_tail, prev_acc_tail = voc[body:], acc[body:]
            s += CH
        if prev_voc_tail is not None and len(prev_voc_tail):
            voc_parts.append(prev_voc_tail)
            acc_parts.append(prev_acc_tail)
        for parts, path in [(voc_parts, voc_p), (acc_parts, acc_p)]:
            full = torch.cat(parts)
            x16 = torchaudio.functional.resample(full, 44100, 16000)
            sf.write(path, x16.clamp(-1, 1).numpy(), 16000, subtype='PCM_16')
        dur = len(wav) / 44100
        print(f'{streamer}/{w.stem}: {dur/60:.0f}min in {time.time()-t0:.0f}s '
              f'({dur/(time.time()-t0):.0f}x RT)', flush=True)
    print('SHARD DONE', flush=True)


if __name__ == '__main__':
    main()
