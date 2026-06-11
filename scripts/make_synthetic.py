"""Synthesize hard training clips from separated stems.

Types (each a homogeneous ~180s clip):
  remix_other  label 0: target streamer's accompaniment + ANOTHER streamer's vocals
               -> confuser (d): "same backing track, different singer"
  accomp_only  label 0: target streamer's accompaniment alone
               -> confusers (a)/(c): karaoke track playing, nobody singing
  voc_mismatch label 1: target streamer's vocals + random unrelated accompaniment
               -> robustness: it's still A singing even if the track doesn't match

Outputs mel npy + constant label npy under data/cache/synth/ + synth_manifest.json.

Usage: python scripts/make_synthetic.py --target 歌回【东雪莲】 --per-type 40 --gpu 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path('/root/Autoslicer')
sys.path.insert(0, str(ROOT / 'src'))
from features import FeatureConfig, LogMel  # noqa: E402

CLIP_S = 180
SR = 16000
OUT_FPS = 3.125


def load_positive_segs(streamer: str) -> list[tuple[str, float, float]]:
    out = []
    for sj in (ROOT / 'data/cache/labels' / streamer).glob('*.segs.json'):
        for s in json.loads(sj.read_text()):
            if s['positive'] and s['end'] - s['start'] >= 60:
                out.append((sj.name.replace('.segs.json', ''), s['start'], s['end']))
    return out


def read_span(streamer: str, stem: str, kind: str, start_s: float, dur_s: float) -> np.ndarray:
    p = ROOT / 'data/cache/stems' / streamer / f'{stem}.{kind}.wav'
    x, _ = sf.read(p, start=int(start_s * SR), frames=int(dur_s * SR), dtype='float32')
    return x


def rms_db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-8)


def sample_span(rng, segs, dur):
    stem, s, e = segs[rng.integers(0, len(segs))]
    if e - s <= dur:
        return stem, s, e - s
    start = float(rng.uniform(s, e - dur))
    return stem, start, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='歌回【东雪莲】')
    ap.add_argument('--per-type', type=int, default=40)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = f'cuda:{args.gpu}'
    frontend = LogMel(FeatureConfig()).to(device)

    streamers = sorted(d.name for d in (ROOT / 'data/cache/stems').iterdir() if d.is_dir())
    others = [s for s in streamers if s != args.target]
    tgt_segs = load_positive_segs(args.target)
    oth_segs = {s: load_positive_segs(s) for s in others}
    oth_segs = {k: v for k, v in oth_segs.items() if v}
    print(f'target segs: {len(tgt_segs)}, others: {[(k, len(v)) for k, v in oth_segs.items()]}', flush=True)
    if not tgt_segs or not oth_segs:
        print('NOT ENOUGH SEGMENTS, aborting')
        return

    out_dir = ROOT / 'data/cache/synth'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(name: str, wav: np.ndarray, label: float):
        wav = np.clip(wav, -1, 1)
        mel = frontend(torch.from_numpy(wav).to(device))[0].cpu().to(torch.float16).numpy()
        n_out = mel.shape[1] // 16
        lab = np.full(n_out, label, dtype=np.float32)
        np.save(out_dir / f'{name}.mel.npy', mel)
        np.save(out_dir / f'{name}.lab.npy', lab)
        sf.write(out_dir / f'{name}.wav', wav, SR, subtype='PCM_16')  # for emb extraction
        manifest.append({'mel': str(out_dir / f'{name}.mel.npy'),
                         'label': str(out_dir / f'{name}.lab.npy'), 'vod': name})

    for i in range(args.per_type):
        # 1) other singer over target's accompaniment (label 0)
        stem_t, s_t, d_t = sample_span(rng, tgt_segs, CLIP_S)
        oname = list(oth_segs)[rng.integers(0, len(oth_segs))]
        stem_o, s_o, d_o = sample_span(rng, oth_segs[oname], CLIP_S)
        dur = min(d_t, d_o)
        acc = read_span(args.target, stem_t, 'accomp', s_t, dur)
        voc = read_span(oname, stem_o, 'vocals', s_o, dur)
        n = min(len(acc), len(voc))
        gain = 10 ** ((rms_db(acc) - rms_db(voc) + rng.uniform(-2, 4)) / 20)
        emit(f'remix_other_{i:03d}', acc[:n] + gain * voc[:n], 0.0)

        # 2) accompaniment only (label 0)
        stem_t, s_t, d_t = sample_span(rng, tgt_segs, CLIP_S)
        emit(f'accomp_only_{i:03d}', read_span(args.target, stem_t, 'accomp', s_t, d_t), 0.0)

        # 3) target vocals over random accompaniment (label 1)
        stem_t, s_t, d_t = sample_span(rng, tgt_segs, CLIP_S)
        any_str = streamers[rng.integers(0, len(streamers))]
        segs2 = tgt_segs if any_str == args.target else oth_segs.get(any_str, tgt_segs)
        stem_a, s_a, d_a = sample_span(rng, segs2, CLIP_S)
        dur = min(d_t, d_a)
        voc = read_span(args.target, stem_t, 'vocals', s_t, dur)
        acc_src = any_str if segs2 is not tgt_segs else args.target
        acc = read_span(acc_src, stem_a, 'accomp', s_a, dur)
        n = min(len(voc), len(acc))
        # scale accomp 2-8 dB below the vocals
        acc_gain = 10 ** ((rms_db(voc) - rms_db(acc) - rng.uniform(2, 8)) / 20)
        emit(f'voc_mismatch_{i:03d}', voc[:n] + acc[:n] * acc_gain, 1.0)
        if (i + 1) % 10 == 0:
            print(f'{i+1}/{args.per_type}', flush=True)

    (out_dir / 'synth_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f'DONE {len(manifest)} clips', flush=True)


if __name__ == '__main__':
    main()
