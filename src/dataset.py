"""Window dataset over cached log-mel features + frame labels.

Each VOD contributes:
  mel:   float16 npy [80, T]        (50 fps)
  label: float32 npy [T_out]        (3.125 fps = 50/16), 1.0 where person A sings

Training samples are random crops of `window_s` seconds with balanced sampling:
`pos_fraction` of crops are forced to overlap a positive region.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def collate_numpy(batch):
    """Stack to numpy in the worker; tensors are created in the main process.
    torch-tensor IPC needs shared mmap which SIGBUSes on this container's
    overlayfs /tmp + 64M /dev/shm; plain numpy pickling over the pipe is safe."""
    xs, ys = zip(*batch)
    return np.stack(xs), np.stack(ys)


@dataclasses.dataclass
class WindowConfig:
    window_s: float = 96.0
    mel_fps: float = 50.0
    out_fps: float = 3.125
    pos_fraction: float = 0.5
    # SpecAugment
    freq_mask: int = 12
    time_mask: int = 60
    n_masks: int = 2
    gain_jitter_db: float = 4.0


class VODWindows(Dataset):
    def __init__(self, pairs: list, cfg: WindowConfig = WindowConfig(),
                 train: bool = True, samples_per_epoch: int = 4000,
                 input_type: str = "mel"):
        """pairs: [(feature_npy, label_npy)] or [(feature_npy, label_npy, focus_regions)].

        input_type="mel": feature is [n_mels, T] at mel_fps; labels at out_fps = mel_fps/16.
        input_type="emb": feature is [T', D] on the SAME grid as labels (ratio 1);
                          augmentation is skipped.
        Labels may contain -1 = ignore (excluded from the loss).
        focus_regions: optional [(start_s, end_s)] regions to oversample (e.g. other
        streamers' singing as hard negatives); used like pos_regions for crop centers.
        """
        self.input_type = input_type
        self.cfg = cfg
        self.train = train
        self.samples_per_epoch = samples_per_epoch
        self.mels, self.labels = [], []
        self.pos_regions = []  # per vod: list of (start_out, end_out) crop-center runs
        for item in pairs:
            mel_p, lab_p = item[0], item[1]
            focus = item[2] if len(item) > 2 else None
            mel = np.load(mel_p, mmap_mode="r")
            lab = np.load(lab_p)
            self.mels.append(mel)
            self.labels.append(lab)
            if focus:
                self.pos_regions.append(
                    [(int(s * cfg.out_fps), max(int(s * cfg.out_fps) + 1, int(e * cfg.out_fps)))
                     for s, e in focus])
            else:
                d = np.diff(np.concatenate([[0], (lab > 0.5).astype(np.int8), [0]]))
                starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
                self.pos_regions.append(list(zip(starts, ends)))
        self.out_win = int(cfg.window_s * cfg.out_fps)
        self.ratio = int(round(cfg.mel_fps / cfg.out_fps)) if input_type == "mel" else 1
        self.mel_win = self.out_win * self.ratio
        self.rng = np.random.default_rng()
        # eval: deterministic tiling of every vod
        self.eval_index = []
        if not train:
            for vi, lab in enumerate(self.labels):
                for o in range(0, max(1, len(lab) - self.out_win + 1), self.out_win):
                    self.eval_index.append((vi, o))

    def __len__(self):
        return self.samples_per_epoch if self.train else len(self.eval_index)

    def _crop(self, vi: int, out_start: int):
        lab = self.labels[vi]
        out_start = int(np.clip(out_start, 0, max(0, len(lab) - self.out_win)))
        mel_start = out_start * self.ratio
        if self.input_type == "mel":
            mel = self.mels[vi][:, mel_start: mel_start + self.mel_win]
        else:
            mel = self.mels[vi][mel_start: mel_start + self.mel_win].T  # [D, T']
        y = lab[out_start: out_start + self.out_win]
        mel = np.asarray(mel, dtype=np.float32)
        if mel.shape[1] < self.mel_win:  # pad tail
            mel = np.pad(mel, ((0, 0), (0, self.mel_win - mel.shape[1])), constant_values=np.log(1e-5))
            y = np.pad(y, (0, self.out_win - len(y)))
        return mel, y.astype(np.float32)

    def _augment(self, mel: np.ndarray) -> np.ndarray:
        c = self.cfg
        mel = mel + np.float32(self.rng.uniform(-c.gain_jitter_db, c.gain_jitter_db) * np.log(10) / 20)
        for _ in range(c.n_masks):
            f0 = self.rng.integers(0, mel.shape[0] - c.freq_mask)
            mel[f0: f0 + self.rng.integers(1, c.freq_mask + 1)] = mel.mean()
            t0 = self.rng.integers(0, max(1, mel.shape[1] - c.time_mask))
            mel[:, t0: t0 + self.rng.integers(1, c.time_mask + 1)] = mel.mean()
        return mel

    def __getitem__(self, i):
        if not self.train:
            vi, o = self.eval_index[i]
            return self._crop(vi, o)
        vi = int(self.rng.integers(0, len(self.mels)))
        lab = self.labels[vi]
        if self.pos_regions[vi] and self.rng.random() < self.cfg.pos_fraction:
            a, b = self.pos_regions[vi][int(self.rng.integers(0, len(self.pos_regions[vi])))]
            center = int(self.rng.integers(a, max(a + 1, b)))
            out_start = center - self.out_win // 2
        else:
            out_start = int(self.rng.integers(0, max(1, len(lab) - self.out_win)))
        mel, y = self._crop(vi, out_start)
        if self.input_type == "mel":
            mel = self._augment(mel)
        return np.ascontiguousarray(mel, dtype=np.float32), y
