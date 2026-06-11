"""Training loop for frame-level singing detection models.

Usage:
  python src/train.py --model crnn --run-name crnn_a --gpu 0 \
      --manifest data/processed/manifest_train.json --val-manifest data/processed/manifest_val.json

Manifest format: [{"mel": "path.npy", "label": "path.npy", "vod": "name"}, ...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from dataset import VODWindows, WindowConfig  # noqa: E402
from metrics import frame_metrics, event_metrics  # noqa: E402
from models import build_model, count_params  # noqa: E402
from postprocess import PostProcessConfig, probs_to_segments, mask_to_runs  # noqa: E402


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pairs(manifest: Path) -> list[tuple[Path, Path]]:
    items = json.loads(Path(manifest).read_text())
    return [(Path(it["mel"]), Path(it["label"])) for it in items]


@torch.no_grad()
def evaluate_full(model, pairs, device, out_fps=3.125, chunk_out=2048, overlap_out=128,
                  input_type="mel") -> dict:
    """Sliding full-VOD inference -> frame & event metrics aggregated over all val VODs."""
    model.eval()
    ratio = 16 if input_type == "mel" else 1
    all_frame = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    ev_pred, ev_true = [], []
    t_offset = 0.0
    for mel_p, lab_p in pairs:
        mel = np.load(mel_p, mmap_mode="r")
        lab = np.load(lab_p)
        n_out = len(lab)
        probs = np.zeros(n_out, dtype=np.float32)
        weight = np.zeros(n_out, dtype=np.float32)
        step = chunk_out - overlap_out
        for o in range(0, n_out, step):
            a, b = o, min(n_out, o + chunk_out)
            if input_type == "mel":
                mel_chunk = np.asarray(mel[:, a * ratio: b * ratio], dtype=np.float32)
            else:
                mel_chunk = np.asarray(mel[a:b], dtype=np.float32)  # [T', D]
            x = torch.from_numpy(mel_chunk)[None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)[0].float().cpu().numpy()
            m = min(len(logits), b - a)
            probs[a: a + m] += 1 / (1 + np.exp(-logits[:m]))
            weight[a: a + m] += 1
            if b >= n_out:
                break
        probs /= np.maximum(weight, 1)
        p = (probs >= 0.5).astype(np.int8)
        t = (lab >= 0.5).astype(np.int8)
        all_frame["tp"] += int(((p == 1) & (t == 1)).sum())
        all_frame["fp"] += int(((p == 1) & (t == 0)).sum())
        all_frame["fn"] += int(((p == 0) & (t == 1)).sum())
        all_frame["tn"] += int(((p == 0) & (t == 0)).sum())
        # events on this vod, offset to keep vods disjoint on a global timeline
        segs = probs_to_segments(probs, PostProcessConfig(frame_rate=out_fps))
        ev_pred += [(s["start"] + t_offset, s["end"] + t_offset) for s in segs]
        for a, b in mask_to_runs(t.astype(bool)):
            ev_true.append((a / out_fps + t_offset, b / out_fps + t_offset))
        t_offset += n_out / out_fps + 3600
    tp, fp, fn, tn = all_frame["tp"], all_frame["fp"], all_frame["fn"], all_frame["tn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"frame_precision": prec, "frame_recall": rec, "frame_f1": f1,
           "frame_acc": (tp + tn) / max(1, tp + fp + fn + tn)}
    out.update(event_metrics(ev_pred, ev_true, iou_threshold=0.5))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="crnn")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--window-s", type=float, default=96.0)
    ap.add_argument("--pos-fraction", type=float, default=0.5)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-kwargs", default="{}")
    ap.add_argument("--input-type", default="mel", choices=["mel", "emb"])
    ap.add_argument("--out-fps", type=float, default=None,
                    help="label frame rate; default 3.125 for mel, 1.5625 for emb")
    args = ap.parse_args()
    if args.out_fps is None:
        args.out_fps = 3.125 if args.input_type == "mel" else 1.5625

    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tb"))
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    wcfg = WindowConfig(window_s=args.window_s, pos_fraction=args.pos_fraction,
                        out_fps=args.out_fps)
    train_pairs = load_pairs(args.manifest)
    val_pairs = load_pairs(args.val_manifest)
    train_ds = VODWindows(train_pairs, wcfg, train=True, input_type=args.input_type,
                          samples_per_epoch=args.batch_size * args.val_every)
    dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)

    model = build_model(args.model, **json.loads(args.model_kwargs)).to(device)
    print(f"model={args.model} params={count_params(model)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, s / args.steps)))))

    best_score, step, t0 = -1.0, 0, time.time()
    history = []
    while step < args.steps:
        for mel, y in dl:
            if step >= args.steps:
                break
            mel, y = mel.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.input_type == "emb":
                mel = mel.transpose(1, 2)  # [B, D, T'] -> [B, T', D]
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mel)
                n = min(logits.shape[1], y.shape[1])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, :n], y[:, :n])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            if step % 200 == 0:
                print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                m = evaluate_full(model, val_pairs, device, out_fps=args.out_fps,
                                  input_type=args.input_type)
                for k, v in m.items():
                    if isinstance(v, (int, float)) and v is not None:
                        writer.add_scalar(f"val/{k}", v, step)
                score = 0.5 * m["frame_f1"] + 0.5 * m["event_f1"]
                history.append({"step": step, **{k: v for k, v in m.items()}, "score": score})
                (run_dir / "history.json").write_text(json.dumps(history, indent=2))
                print(f"[val @{step}] frame_f1={m['frame_f1']:.4f} event_f1={m['event_f1']:.4f} "
                      f"P/R={m['frame_precision']:.3f}/{m['frame_recall']:.3f} score={score:.4f}", flush=True)
                torch.save({"model": model.state_dict(), "step": step, "metrics": m,
                            "args": vars(args)}, run_dir / "last.pt")
                if score > best_score:
                    best_score = score
                    torch.save({"model": model.state_dict(), "step": step, "metrics": m,
                                "args": vars(args)}, run_dir / "best.pt")
                    print(f"  new best score={score:.4f}", flush=True)
    print(f"DONE best_score={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
