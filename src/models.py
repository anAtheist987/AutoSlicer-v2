"""Model zoo for AutoSlicer v2.

All models map log-mel [B, n_mels, T] -> frame logits [B, T // time_pool].
Positive class = "person A is singing right now".
"""
from __future__ import annotations

import torch
from torch import nn


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class ConvBlock(nn.Module):
    """2x (conv3x3-BN-GELU) + pooling, PANNs-style."""

    def __init__(self, in_ch: int, out_ch: int, pool_t: int = 2, pool_f: int = 2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )
        self.pool = nn.AvgPool2d((pool_f, pool_t)) if (pool_f > 1 or pool_t > 1) else nn.Identity()

    def forward(self, x):
        return self.pool(self.body(x))


class MelCRNN(nn.Module):
    """Path A: small CRNN trained from scratch.

    log-mel [B, 80, T] -> CNN (time/16, freq/16) -> BiGRU -> logits [B, T/16].
    ~4M params at default width.
    """

    def __init__(self, n_mels: int = 80, channels=(32, 64, 128, 256),
                 rnn_hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_mels)
        blocks, in_ch = [], 1
        for ch in channels:
            blocks.append(ConvBlock(in_ch, ch, pool_t=2, pool_f=2))
            in_ch = ch
        self.cnn = nn.Sequential(*blocks)
        feat_dim = channels[-1] * (n_mels // 2 ** len(channels))
        self.proj = nn.Sequential(nn.Linear(feat_dim, rnn_hidden), nn.LayerNorm(rnn_hidden), nn.GELU())
        self.rnn = nn.GRU(rnn_hidden, rnn_hidden // 2, num_layers=rnn_layers,
                          bidirectional=True, batch_first=True,
                          dropout=dropout if rnn_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(rnn_hidden, 1))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.bn0(mel)[:, None]                  # [B, 1, F, T]
        x = self.cnn(x)                             # [B, C, F', T']
        x = x.permute(0, 3, 1, 2).flatten(2)        # [B, T', C*F']
        x = self.proj(x)
        x, _ = self.rnn(x)
        return self.head(x)[..., 0]                 # [B, T']


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.body(x)


class MelTCN(nn.Module):
    """Path A': CNN front + dilated TCN context (no recurrence — fully parallel,
    receptive field ~ +/-1100 frames at 3.125fps ~= +/-6 min with default dilations)."""

    def __init__(self, n_mels: int = 80, channels=(32, 64, 128, 256),
                 tcn_ch: int = 256, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256), dropout: float = 0.2):
        super().__init__()
        self.bn0 = nn.BatchNorm1d(n_mels)
        blocks, in_ch = [], 1
        for ch in channels:
            blocks.append(ConvBlock(in_ch, ch, pool_t=2, pool_f=2))
            in_ch = ch
        self.cnn = nn.Sequential(*blocks)
        feat_dim = channels[-1] * (n_mels // 2 ** len(channels))
        self.proj = nn.Conv1d(feat_dim, tcn_ch, 1)
        self.tcn = nn.Sequential(*[TCNBlock(tcn_ch, d, dropout=dropout) for d in dilations])
        self.head = nn.Conv1d(tcn_ch, 1, 1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.bn0(mel)[:, None]
        x = self.cnn(x)                             # [B, C, F', T']
        x = x.permute(0, 1, 3, 2).flatten(1, -2) if False else x.permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
        # x: [B, C*F', T']
        x = self.proj(x)
        x = self.tcn(x)
        return self.head(x)[:, 0]                   # [B, T']


class BackboneHead(nn.Module):
    """Path B: frozen/pretrained embedding sequence [B, T', D] -> BiGRU -> logits.
    The backbone runs separately (cached embeddings); this is just the trainable head."""

    def __init__(self, in_dim: int, hidden: int = 256, rnn_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU())
        self.rnn = nn.GRU(hidden, hidden // 2, num_layers=rnn_layers, bidirectional=True,
                          batch_first=True, dropout=dropout if rnn_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(emb)
        x, _ = self.rnn(x)
        return self.head(x)[..., 0]


def build_model(name: str, **kw) -> nn.Module:
    name = name.lower()
    if name == "crnn":
        return MelCRNN(**kw)
    if name == "tcn":
        return MelTCN(**kw)
    if name == "backbone_head":
        return BackboneHead(**kw)
    raise ValueError(f"unknown model {name}")


if __name__ == "__main__":
    for name in ["crnn", "tcn"]:
        m = build_model(name)
        x = torch.randn(2, 80, 3000)  # 60 s
        y = m(x)
        print(f"{name}: params={count_params(m)/1e6:.2f}M out={tuple(y.shape)}")
