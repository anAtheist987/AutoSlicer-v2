"""Log-mel feature frontend.

All models share one frontend so cached features can be reused:
  16 kHz mono -> STFT(n_fft=1024, hop=320) -> 80 mel bands -> log.
Frame rate = 50 fps; CNN backbones further pool time by 16x -> 3.125 fps
(0.32 s per output frame), which is plenty for song-boundary precision.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import torchaudio


@dataclasses.dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 320          # 20 ms -> 50 fps
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 8000.0
    time_pool: int = 16            # CNN time pooling; output fps = 50/16 = 3.125

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def output_frame_rate(self) -> float:
        return self.frame_rate / self.time_pool


class LogMel(torch.nn.Module):
    def __init__(self, cfg: FeatureConfig = FeatureConfig()):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2.0,
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, L] or [L] -> log-mel [B, n_mels, T]"""
        if wav.dim() == 1:
            wav = wav[None]
        m = self.mel(wav)
        return torch.log(m + 1e-5)


def load_audio(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    """Load any audio/video file to mono float32 at sample_rate. Uses ffmpeg for robustness."""
    path = str(path)
    try:
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        return wav[0]
    except Exception:
        # fall back to ffmpeg piping (handles video containers, m4a, etc.)
        import subprocess
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
               "-ac", "1", "-ar", str(sample_rate), "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return torch.frombuffer(bytearray(raw), dtype=torch.float32)


def extract_logmel_file(
    path: str | Path, cfg: FeatureConfig = FeatureConfig(),
    device: str = "cpu", chunk_seconds: int = 600,
) -> np.ndarray:
    """Whole-file log-mel as float16 numpy [n_mels, T]; chunked to bound memory."""
    frontend = LogMel(cfg).to(device)
    wav = load_audio(path, cfg.sample_rate)
    hop = cfg.hop_length
    chunk = chunk_seconds * cfg.sample_rate
    chunk = (chunk // hop) * hop
    outs = []
    for s in range(0, len(wav), chunk):
        seg = wav[s: s + chunk].to(device)
        if len(seg) < cfg.n_fft:
            break
        outs.append(frontend(seg)[0].cpu().to(torch.float16).numpy())
    return np.concatenate(outs, axis=1)


def cache_features(
    audio_paths: list[Path], cache_dir: Path,
    cfg: FeatureConfig = FeatureConfig(), device: str = "cpu",
) -> dict[str, Path]:
    """Extract+cache log-mel for each file. Returns {stem: npy_path}."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for p in audio_paths:
        npy = cache_dir / (Path(p).stem + ".npy")
        if not npy.exists():
            feat = extract_logmel_file(p, cfg, device)
            np.save(npy, feat)
        out[Path(p).stem] = npy
    return out
