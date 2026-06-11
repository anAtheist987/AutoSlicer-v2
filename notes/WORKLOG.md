# AutoSlicer v2 工作日志

任务：识别主播A唱歌片段并自动切片成独立视频。截止：2026-06-12 09:30。

## 时间线

- **01:45** 环境确认：7×RTX 3090，GPU 0/1/2/4 空闲（3/5/6 有他人进程，不可动）。
  torch 2.6.0+cu124。已装 librosa/soundfile/torchaudio/ffmpeg。
  网络：huggingface.co 被墙 → 用 hf-mirror.com；GitHub/Zenodo 可用。
- **01:46** 读完旧项目 `/root/AutoSlicer`：8kHz 波形 → SincConv + ResNeXt1d 塔（729× 降采样）
  → BiGRU → 逐帧 BCE，wav2vec 式对比预训练。痛点：感受野不足、需数分钟上下文、规模小。
- **01:50** 启动调研 workflow（6 方向并行：SVD 文献 / 现代骨干 / 生产 VAD / 歌手验证 /
  源分离 / 后处理+长上下文）→ 输出 notes/RESEARCH.md。
- **01:50–02:00** 数据集 zip 仍在向 /root/AgentGateway/ 传输（~3MB/s），挂了自动监视+解压任务
  → data/raw/。zip 内见 `convert/` 目录。
- **02:00** 写好共享代码骨架（与数据格式解耦）：
  - `src/features.py` — 16kHz / 80-mel / hop 320 (50fps)，CNN 再 16× 时间池化 → 输出 3.125fps
  - `src/models.py` — 路径A `MelCRNN`(2.1M)、路径A' `MelTCN`(3.9M, 膨胀卷积感受野 ±6min)、
    路径B `BackboneHead`（预训练嵌入 + BiGRU 头）
  - `src/dataset.py` — 缓存 mel 的窗口采样（正例平衡 + SpecAugment）
  - `src/train.py` — AMP bf16、cosine+warmup、全 VOD 滑窗验证（frame-F1 + event-F1）、最优保存
  - `src/postprocess.py` — 中值平滑 → 双阈值滞回 → 去短血点 → 间奏填补(≤45s) →
    邻段合并(≤90s，处理"唱1分钟-放别人版本1分钟-再唱"→一段) → 最短歌长 → 切片 padding
  - `src/metrics.py` — frame P/R/F1 + IoU≥0.5 事件级 F1 + 边界 MAE
  - `src/infer.py` — 视频/音频 → ffmpeg 解码 → 滑窗推理 → 切点 JSON → ffmpeg -c copy 快剪
- **02:00** 预训练权重：PANNs CNN14_16k (Zenodo, 358MB) ✓；EfficientAT mn10_as 下载中。

## 决策记录

1. **共享 mel 前端**（16kHz/80mel/50fps）：所有路径共用缓存特征，省去重复抽取；
   输出帧率 3.125fps（0.32s 分辨率），对歌曲级边界足够。
2. **三条候选路径**：A=从零 CRNN（基线，必然能训出来）；A'=TCN（无循环、全并行、长感受野）；
   B=PANNs/EfficientAT 预训练嵌入 + 轻头（少量数据下泛化更稳）。
   C(可选)=ECAPA 歌手验证分支处理"放别人唱的同一首歌"干扰。视训练情况取舍。
3. **验证协议**：按 VOD 划分 train/val（绝不按帧随机分），主指标 = 0.5*frame-F1 + 0.5*event-F1。
4. **合并语义**：按用户要求，中间插放他人版本 ≤90s 时整体算一段（PostProcessConfig.merge_gap_s）。

- **02:45–03:10** 关键转折：zip 传输近停滞、PANNs 在 8kHz 窄带上唱歌检测退化（p99=0.38）
  → 启动**自监督数据工厂**主路线（见 DESIGN.md）：HTDemucs 3-GPU 分块分离（修复了整文件载入
  OOM 静默死亡问题）→ YIN/能量伪标签（实测：聊天候选段被 sustained_frac 过滤器正确拒绝，
  唱歌段呈 1.3-3.9min 歌曲形态）→ 其他主播=声纹难负例 → stem 重混合成干扰(d)样本。
- **03:10** 执行 workflow 启动（run wf_3c3d5289-65e）：Prep(等分离+伪标签+合成+manifests) →
  Train(crnn@GPU0, tcn@GPU1, panns_head@GPU2 并行) → Tune(后处理网格) → Demo(端到端切片+RTF)。
  另设 18min 周期巡检 cron 监控卡死与 zip 标注到达。
