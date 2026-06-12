# AutoSlicer v2

为主播歌回直播自动切片设计的歌声检测系统：识别**特定主播本人在唱歌**的片段，
把每首歌剪成独立视频。v2 完全重写（2026-06），替代旧版 8kHz SincConv+ResNeXt+BiGRU 方案。

**最终报告（指标/设计/实验全记录）→ [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)**

## 核心数字

- 人工金标验证：事件级 F1 0.83（P 0.88 / R 0.79），切点误差 ~6s，帧级 F1 0.89
- 声纹判别：他人 2.1h 歌回 0 误报（解决"放别人唱的同一首歌"干扰）
- 速度：2.5h VOD 端到端 15 秒（RTF 0.0005），显存 <2GB，模型 2.1M 参数

## 快速使用

```bash
python src/infer.py --input vod.mp4 --checkpoint runs/gold_crnn/best.pt \
    --out-dir out/ --gpu 0 --on-threshold 0.6 --off-threshold 0.3 --cut-video
# -> out/vod.segments.json + out/vod_songNN.mp4 (ffmpeg -c copy 秒级切片)
```

## 训练新主播

有 Audition 标注（Name/Start/Duration 的 tab 分隔 csv）：参考 `scripts/gold_azusa.py`。
无标注（自监督冷启动）：
```bash
python scripts/separate_all.py --gpu 0          # HTDemucs 分离 vocals/accomp
python scripts/pseudo_label.py                  # YIN+能量 伪标签
python scripts/make_synthetic.py --target 歌回【主播名】   # stem 重混难样本
python scripts/build_manifests.py --target 歌回【主播名】
python src/train.py --model crnn --run-name my_run --gpu 0 \
    --manifest data/processed/manifest_train.json --val-manifest data/processed/manifest_val.json
python scripts/tune_postprocess.py --probs-dir runs/my_run/val_probs
```

## 目录

| 路径 | 内容 |
|---|---|
| `src/` | features / models / dataset / train / infer / postprocess / metrics |
| `scripts/` | 数据工厂（解压/分离/伪标签/合成/嵌入/manifests/金标/调参） |
| `notes/` | RESEARCH.md（SOTA 调研）/ DESIGN.md / WORKLOG.md |
| `reports/` | FINAL_REPORT.md |
| `runs/` | 训练产物（gitignore，`gold_crnn/best.pt` 为最终模型） |
| `data/` | 数据与缓存（gitignore） |
