"""PANNs Cnn14_16k framewise embedding extractor (Path B backbone).

Compact re-implementation of Cnn14 from qiuqiangkong/audioset_tagging_cnn (MIT
license) sufficient to load the released `Cnn14_16k_mAP=0.438.pth` checkpoint and
emit framewise embeddings BEFORE clip pooling:

  wav 16k -> logmel (win 512, hop 160, 64 mel) @100 fps
          -> 6 conv blocks, each avgpool 2x2  -> [B, 2048, T/64]  @1.5625 fps

Embeddings are cached as float16 npy [T', 2048] per VOD.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

PANNS_FPS = 100.0 / 64.0  # 1.5625 embedding frames per second
EMB_DIM = 2048


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool_size)


class Cnn14_16k(nn.Module):
    def __init__(self):
        super().__init__()
        sr, n_fft, hop, mels, fmin, fmax = 16000, 512, 160, 64, 50, 8000
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann",
            center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=sr, n_fft=n_fft, n_mels=mels, fmin=fmin, fmax=fmax,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        # present in the checkpoint; unused for framewise features but kept so
        # load_state_dict(strict=True) validates the full file
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, L] -> framewise embedding [B, T', 2048] @1.5625 fps"""
        x = self.spectrogram_extractor(wav)          # [B, 1, T, freq]
        x = self.logmel_extractor(x)                 # [B, 1, T, 64]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        for block in [self.conv_block1, self.conv_block2, self.conv_block3,
                      self.conv_block4, self.conv_block5, self.conv_block6]:
            x = block(x, pool_size=(2, 2))
        x = torch.mean(x, dim=3)                     # [B, 2048, T']
        x1 = F.max_pool1d(x, 3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, 3, stride=1, padding=1)
        x = x1 + x2
        return x.transpose(1, 2)                     # [B, T', 2048]


def load_panns(checkpoint: str | Path, device: str = "cpu") -> Cnn14_16k:
    model = Cnn14_16k()
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def extract_embeddings_file(path: str | Path, model: Cnn14_16k, device: str,
                            chunk_s: float = 300.0, overlap_s: float = 0.0) -> np.ndarray:
    """Whole file -> [T', 2048] float16 at 1.5625 fps."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from features import load_audio
    wav = load_audio(path, 16000)
    sr = 16000
    hop_emb = int(chunk_s * PANNS_FPS)
    chunk = int(chunk_s * sr)
    outs = []
    for s in range(0, len(wav), chunk):
        seg = wav[s: s + chunk]
        if len(seg) < sr:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            emb = model(seg[None].to(device))[0]
        outs.append(emb.float().cpu().to(torch.float16).numpy()[:hop_emb])
    return np.concatenate(outs, axis=0)


if __name__ == "__main__":
    m = load_panns("pretrained/Cnn14_16k_mAP=0.438.pth")
    wav = torch.randn(1, 16000 * 10)
    emb = m(wav)
    print("ok, emb shape for 10s:", tuple(emb.shape))
