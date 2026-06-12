# AutoSlicer v2

为主播歌回直播自动切片设计的歌声检测系统：识别**特定主播本人在唱歌**的片段，
把每首歌剪成独立视频。v2 完全重写（2026-06），替代旧版 8kHz SincConv+ResNeXt+BiGRU 方案。

**最终报告（指标/设计/实验全记录）→ [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)**

## 核心数字

- **独立测试集（未参与训练/调参的 2 场 VOD）：事件级 F1 0.90**（P 0.91 / R 0.89），帧级 F1 0.94
- 3 折交叉验证（6 场金标留二法）：事件级 F1 **0.93 ± 0.03**，帧级 0.956 ± 0.007；换种子重训完全复现
- 声纹判别：他人 2.1h 歌回 0 误报（解决"放别人唱的同一首歌"干扰）
- 速度：2.5h VOD 端到端 15 秒（RTF 0.0005），显存 <2GB，模型 2.1M 参数

## 快速使用

```bash
# 本地文件
python src/infer.py --input vod.mp4 --checkpoint runs/v2_final/best.pt \
    --out-dir out/ --gpu 0 --cut-video
# -> out/vod.segments.json + out/vod_songNN.mp4 (ffmpeg -c copy 秒级切片)

# 直接给 B 站链接（BV 号 / 完整 URL / b23.tv 短链均可），自动下载后切片
python src/infer.py --input "https://www.bilibili.com/video/BVxxxxxxxxxx" \
    --checkpoint runs/v2_final/best.pt --out-dir out/ --cut-video
# 多 P 用 --page N；无登录视频最高 480p，--cookie "SESSDATA=..." 可提清晰度

# 只用下载器
python src/bilibili.py "https://b23.tv/xxxxxx" --out-dir downloads [--video]
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
| `runs/` | 训练产物（gitignore，`v2_final/best.pt` 为最终模型） |
| `data/` | 数据与缓存（gitignore） |

## 安装

```bash
pip install -r requirements.txt   # 推理只需前 5 行；ffmpeg 需在 PATH
```

## 许可证与声明

代码以 [MIT](LICENSE) 发布。训练数据（主播 VOD）不随仓库分发。
B 站下载模块仅供个人学习研究使用：请遵守 bilibili 服务条款，尊重主播对其内容的权利,
切片二次发布前请确认获得授权。依赖项许可:PyTorch (BSD)、Demucs (MIT,仅训练数据
制备)、librosa (ISC)。
