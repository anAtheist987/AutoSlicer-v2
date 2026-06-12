"""Turn per-frame probabilities into song-level cut segments.

Pipeline:
  probs (output_frame_rate Hz)
    -> median smoothing
    -> hysteresis double-threshold (on/off)
    -> drop short positive blips, fill short gaps (intra-song breaks, e.g. instrument solos
       or a short reference playback in the middle of a performance)
    -> merge segments separated by < merge_gap_s (the "sang 1 min, played someone else's
       version 1 min, sang again" rule -> one segment)
    -> drop segments shorter than min_song_s
    -> pad with lead-in / lead-out for the video cut
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import median_filter


@dataclasses.dataclass
class PostProcessConfig:
    frame_rate: float = 3.125      # Hz of the prob sequence
    median_s: float = 3.0          # median filter window
    on_threshold: float = 0.6      # hysteresis: enter singing
    off_threshold: float = 0.4     # hysteresis: leave singing
    min_blip_s: float = 8.0        # positive runs shorter than this are noise
    max_gap_s: float = 45.0        # negative gaps shorter than this inside a song are filled
    merge_gap_s: float = 90.0      # adjacent songs closer than this merge into one cut
    min_song_s: float = 50.0       # discard "songs" shorter than this
    pad_before_s: float = 5.0      # video lead-in
    pad_after_s: float = 5.0       # video lead-out
    # boundary refinement: trim the low-confidence "shoulders" the loose
    # off_threshold leaves at segment edges (pre-song humming, talk over the
    # intro). Only the outermost boundaries move; interior gap-fills stay.
    refine_edges: bool = False
    edge_threshold: float = 0.45   # walk outward from the core while p >= this
    core_threshold: float = 0.60   # "surely singing" level anchoring a boundary
    core_min_s: float = 4.0        # min sustained core run to count as anchor


def smooth(probs: np.ndarray, cfg: PostProcessConfig) -> np.ndarray:
    k = max(1, int(round(cfg.median_s * cfg.frame_rate)) | 1)
    return median_filter(probs.astype(np.float32), size=k)


def hysteresis(probs: np.ndarray, on: float, off: float) -> np.ndarray:
    """Binary mask with double threshold (less flicker than single threshold)."""
    mask = np.zeros(len(probs), dtype=bool)
    active = False
    for i, p in enumerate(probs):
        if not active and p >= on:
            active = True
        elif active and p < off:
            active = False
        mask[i] = active
    # backward pass to also catch runs that start above `off` but only later cross `on`
    active = False
    for i in range(len(probs) - 1, -1, -1):
        if not active and probs[i] >= on:
            active = True
        elif active and probs[i] < off:
            active = False
        mask[i] |= active
    return mask


def mask_to_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(idx[0::2], idx[1::2]))


def refine_run(p: np.ndarray, a: int, b: int, cfg: PostProcessConfig) -> tuple[int, int]:
    """Re-place [a, b)'s outer boundaries: anchor on the first/last sustained
    high-confidence core run, then extend outward while p >= edge_threshold.
    Falls back to the original boundary if no core exists or the trim would
    drop the segment below min_song_s."""
    fr = cfg.frame_rate
    k = max(1, int(round(cfg.core_min_s * fr)))
    core = p[a:b] >= cfg.core_threshold
    runs = [(ra, rb) for ra, rb in mask_to_runs(core) if rb - ra >= k]
    if not runs:
        return a, b
    na = a + runs[0][0]
    while na > a and p[na - 1] >= cfg.edge_threshold:
        na -= 1
    nb = a + runs[-1][1]
    while nb < b and p[nb] >= cfg.edge_threshold:
        nb += 1
    if (nb - na) / fr < cfg.min_song_s:
        return a, b
    return na, nb


def probs_to_segments(probs: np.ndarray, cfg: PostProcessConfig = PostProcessConfig(),
                      total_duration_s: float | None = None) -> list[dict]:
    """Returns [{'start': s, 'end': s, 'cut_start': s, 'cut_end': s}] in seconds."""
    fr = cfg.frame_rate
    p = smooth(probs, cfg)
    mask = hysteresis(p, cfg.on_threshold, cfg.off_threshold)

    runs = mask_to_runs(mask)
    runs = [(a, b) for a, b in runs if (b - a) / fr >= cfg.min_blip_s]

    # fill short gaps (intra-song)
    filled: list[list[int]] = []
    for a, b in runs:
        if filled and (a - filled[-1][1]) / fr <= cfg.max_gap_s:
            filled[-1][1] = b
        else:
            filled.append([a, b])

    # merge near-adjacent songs into one cut
    merged: list[list[int]] = []
    for a, b in filled:
        if merged and (a - merged[-1][1]) / fr <= cfg.merge_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    merged = [(a, b) for a, b in merged if (b - a) / fr >= cfg.min_song_s]

    if cfg.refine_edges:
        merged = [refine_run(p, a, b, cfg) for a, b in merged]

    out = []
    for a, b in merged:
        s, e = a / fr, b / fr
        cs = max(0.0, s - cfg.pad_before_s)
        ce = e + cfg.pad_after_s
        if total_duration_s is not None:
            ce = min(ce, total_duration_s)
        out.append({"start": round(s, 2), "end": round(e, 2),
                    "cut_start": round(cs, 2), "cut_end": round(ce, 2)})
    return out


def segments_to_mask(segments: list[tuple[float, float]], n_frames: int, frame_rate: float) -> np.ndarray:
    """Ground-truth (start_s, end_s) list -> frame mask."""
    m = np.zeros(n_frames, dtype=np.float32)
    for s, e in segments:
        a = int(round(s * frame_rate))
        b = int(round(e * frame_rate))
        m[max(0, a): min(n_frames, b)] = 1.0
    return m
