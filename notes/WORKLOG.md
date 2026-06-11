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
- **03:25 巡检** 分离 22/39（按计划 ~04:00 完）；zip 传输恢复至 ~3MB/s（已 11.1GB），新到 4.3GB
  猫雷大文件已部分提取，暂未见标注 csv；执行 workflow Prep agent 正常轮询中。
- **03:45 巡检** 分离 34/39；zip 已 14.3GB 仍在传，data_8k 之后暂未出现标注目录；
  部分提取持续后台跑。workflow Prep 等待分离收尾。
- **04:00 巡检** 分离全部完成（39 场，~85h，平均 31×RT）。zip 17.9GB 仍在传。
  Prep agent 接管：extract→伪标签→合成→emb→manifests，预计 04:20 起训。
  （处理：停掉了与 prep agent 重复的 extract_partial 后台任务，避免并发写同一文件。）
- **04:10 巡检** 修复 extract_partial 的 ZIP64 解析 bug（4GB+ 条目截断+遍历卡死），
  解出 111 个新条目：顶层 歌回【主播】/*.m4s = 同批 VOD 的全频带原始音频；新增两位主播
  （神楽七奈、红晓音Akane）；data/raw/new/ 出现 wav+Audition pkf（人工标注会话素材，
  marker csv 可能尚在传输）。Prep agent 已进入伪标签阶段（29/39）。
- **04:30 巡检** Prep 全部完成：39 伪标签 + 120 合成片段 + emb 159 + manifests
  (train 154 项 / val 5 项 = 东雪莲×2 + 其他主播×3)。三训练已起：crnn 0.31s/step、
  tcn 0.30s/step (GPU0/1)、panns_head 0.57s/step (GPU2)。预计 05:30-06:10 完成。
- **04:40 巡检** 训练健康推进：tcn@1000 val frame_f1=0.40 (P0.27/R0.78，早期过召回正常)；
  crnn@1000 验证中；panns_head 较慢 (0.6s/step)。训练 agent 自行修复了 eval 解包 3 元组的
  bug（focus 字段引入）。zip 25.1GB 仍在传。
- **04:55 巡检** 中期指标（对伪标签）：tcn@3000 frame0.78/event0.65（最佳 score 0.72）；
  crnn@2000 0.76/0.60；panns_head@1000 0.73/0.62（P=0.99 嵌入信息量足）。
  中期 P/R 振荡属正常，cosine 末段会收敛。zip 28GB。
- **05:03 巡检** crnn@5000 frame0.90/event0.84(score0.87)；tcn@5000 0.88/0.80；
  panns_head 召回卡 0.58。zip 新解出 歌回【阿梓从小就很可爱】（旧项目的标注主播！）
  —— 阿梓的人工标注很可能随后到达。
- **05:21 巡检** tcn@9000 frame0.955/event0.979(score0.967，伪标签val)；crnn@8000 frame0.94；
  panns_head 过拟合(loss≈0)召回卡0.53，预计淘汰。zip 继续解出新主播(露娜/露米)，32.6GB。
  训练即将收尾进入 Tune。
- **05:45 转折点** zip 传输完成(35.4GB)！金标到达：阿梓 标注/ = 7 csv(Audition 标记) + wav，
  6 场可用共 167 首人工标注歌曲(~21h, 正例~50%)。已建金标管线 (scripts/gold_azusa.py)：
  4 场训练 + 2 场验证(47首)，负例复用其他主播(含东雪莲)伪唱歌段 focus。
  gold_crnn@GPU0 / gold_tcn@GPU1 已开训(8k步,~06:35 完)。东雪莲伪标签模型转为
  辅助产物(声纹判别诊断用)。1237149092 的 csv 为空已剔除。
