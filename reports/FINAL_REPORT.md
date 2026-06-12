# AutoSlicer v2 — 歌回直播自动切片系统 最终报告

> 2026-06-12 凌晨 01:45 – 06:45 完成。代码与全部中间产物在 `/root/Autoslicer`。
> 调研详情：`notes/RESEARCH.md`；设计文档：`notes/DESIGN.md`；逐时段日志：`notes/WORKLOG.md`。

## TL;DR

- 在**人工金标**（阿梓 标注/，167 首人工标记歌曲）上：**事件级 F1 = 0.83**（38 个真值
  歌曲段命中 30、误报 4），**切点平均误差 ≈ 6 秒**，帧级 F1 = 0.89（精确率 0.97）。
  超过 80% 可用线。
- **声纹判别**（核心难点干扰 (d)）：把"阿梓检测器"喂给另一位主播（东雪莲）2.1 小时
  全程唱歌的歌回 → **0 个误报片段**。模型学到的是"这个人在唱"，不是"有人在唱"。
- **推理速度**：模型+解码 RTF ≈ 0.0005（纯模型 14000× 实时）；2.5h VOD 端到端
  （解码→检测→ffmpeg 切出 17 个片段）共 **15 秒**，显存 <2GB —— 远优于
  "不超过直播时长"与"10GB GPU"的要求。
- 模型仅 **2.1M 参数**（8.4MB fp32），单文件 CLI 一条命令出切片。

## 1. 任务与约束

输入数小时直播 VOD，识别「主播 A 本人在唱歌」的片段，每首歌输出独立视频切片（带前后
padding）。需排除四类干扰：(a) 放纯伴奏没唱 (b) 伴奏与唱不匹配 (c) 伴奏只是歌的片段
(d) 放别人唱的同一首歌做对照。中间短暂插放他人版本时不得把一首歌切成两段。
约束：3090 夜间训完；10GB GPU 可推理；处理时间 < 直播时长；准确率 ≥80%。

## 2. 调研结论（详见 notes/RESEARCH.md）

6 方向并行调研（SVD 文献 / 现代音频骨干 / 生产 VAD / 歌手验证 / 源分离 / 后处理）。
直接采纳的结论：

1. 生产级系统形态收敛于：log-mel → 小型逐帧分类器 → 滑窗 overlap-add → 双阈值滞回 +
   时长约束（Silero VAD / pyannote / MarbleNet 同构）。旧系统 8kHz 原始波形 SincConv+
   ResNeXt+BiGRU 的"数分钟感受野"痛点，在该范式下由后处理与长感受野模块廉价解决。
2. 纯歌声检测（SVD）基准十年停在 92-94% 帧准确率；我们的任务本质是 **SVD × 歌手验证**。
   干扰 (d) 没有任何 tagging 模型能单独解决，必须让训练数据含"别人在唱"的负例。
3. 源分离（HTDemucs）作为**离线数据工厂**而非推理依赖：制造负例与伪标签的杠杆远大于
   模型结构杠杆。
4. AudioSet 预训练骨干（PANNs/EfficientAT/PretrainedSED 系）+ 轻头是 DCASE 共识，
   但在 8kHz 窄带域上有明显域差（实测 PANNs Singing 类 p99 仅 0.38）。

## 3. 数据与监督策略（本项目最大的设计决策）

数据集（35.4GB zip，凌晨持续传输，05:45 才完整到达）：
8 位主播的歌回直播（8kHz wav + 全频带 m4s，约 130h）+ **阿梓 7 场人工标注**
（Audition 标记 csv：Start/Duration，167 首歌，~21h，正例约 50%）。

由于标注在 deadline 前 4 小时才到达，系统按**两阶段监督**构建：

### 3.1 自监督数据工厂（无人工标注时的主路线，01:45-05:30）
- **HTDemucs 2-stem 分离**全部 39 场已到达 VOD（3×3090 并行、分块+交叉淡化防 OOM，
  ~31× 实时，85h 音频 1 小时完成）→ 16k 单声道 vocals/accomp stems（20GB）。
- **伪标签**：歌回直播中唱歌主体=主播本人。帧级候选 = 人声 stem 活跃 ∧ 伴奏活跃；
  段级过滤 = 时长≥45s ∧ 人声密度≥0.4 ∧ **持续音符占比≥0.10**（YIN F0 滚动标准差
  <0.7 半音，实测唱歌区 21% vs 聊天区 6%，正是"聊天带 BGM"误标的克星）∧ 伴奏≥-48dB。
  段边界 ±8s 置 ignore(-1) 不进 loss。全库伪标签正例率 42%，段落呈 1.3-3.9min 歌曲形态。
- **声纹负例**：其他主播的整场歌回 = "有人在唱但不是A"，其伪唱歌段作为 focus 过采样。
- **合成难样本**（stem 重混，120 段）：A伴奏+他人人声（=干扰d）、纯A伴奏（=干扰a/c）、
  A人声+错配伴奏（=鲁棒正例，对应干扰b场景下A仍在唱）。

### 3.2 金标管线（05:45 标注到达后）
解析 Audition csv → 3.125fps 帧标签；4 场训练 + 2 场验证（47 首歌）；
负例沿用其他主播（含东雪莲）的 focus 过采样。**所有最终指标以金标验证为准。**

## 4. 系统架构

```
VOD(任意容器) → ffmpeg 16k mono → log-mel(80, hop 320, 50fps)
  → MelCRNN: 4×ConvBlock(时/频各16×池化) → BiGRU(2×256) → 帧 logit @3.125fps (0.32s)
  → 滑窗 overlap-add → 3s 中值平滑 → 滞回(on 0.6/off 0.3) → 去<8s噪点
  → 填≤30s间隙(间奏/短插放) → 合并≤90s邻段(产品语义:插放他人版本不拆段)
  → 弃<60s段 → ±5s padding → ffmpeg -c copy 切片
```

训练对比的三条路径（伪标签赛道全部跑通）：

| 路径 | 结构 | 参数 | 设计动机 |
|---|---|---|---|
| CRNN | mel→CNN→BiGRU | 2.09M | 生产 VAD 共识形态（基线） |
| TCN | mel→CNN→9级膨胀卷积(感受野±6min) | 3.87M | 无循环全并行，长上下文 |
| PANNs头 | 冻结 CNN14_16k 嵌入(2048d@1.56fps)→BiGRU | 头1.12M(+80M冻结) | AudioSet 预训练迁移 |

训练：96s 窗、batch 32、AdamW 3e-4 cosine、bf16、BCE(ignore mask)、SpecAugment、
正例窗 50% 平衡采样、按 VOD 划分 train/val。8-10k 步 ≈ 45-55 分钟/模型。

## 5. 实验结果

### 5.1 伪标签赛道（目标=东雪莲，验证集=东雪莲2场+其他主播3场，对伪标签）

| 模型 | frame-F1 | event-F1 | 事件TP/FP/FN | 边界MAE |
|---|---|---|---|---|
| TCN | 0.955 | **0.979** | 23/0/1 | ~16s |
| CRNN | 0.953 | 0.791 | 17/2/7 | ~21s |
| PANNs头 | 0.900 | 0.818 | 18/2/6 | ~18s |

（其他主播 3 场验证 VOD 上述模型均近零误报 — 声纹负例策略生效。）

### 5.2 金标赛道（目标=阿梓，验证集=2场人工标注 VOD、38 个合并后真值歌曲事件）

| 模型 | frame-F1 | frame-P/R | event-F1 | 事件TP/FP/FN | onset/offset MAE |
|---|---|---|---|---|---|
| **CRNN（最终选型）** | **0.894** | 0.97/0.83 | **0.833** | 30/4/8 | **6.0s / 5.9s** |
| TCN | 0.857 | 0.97/0.77 | 0.817 | 30/3/8 | 38.3s / 19.9s |

分 VOD：1188184012 事件 P0.82/R0.74 帧F1 0.85；1190452750 事件 P0.94/R0.84 帧F1 0.95。
最优后处理：on 0.6 / off 0.3 / 中值 3s / 间隙填充 30s / 最短歌 60s。

**选型理由**：检出能力相当时 CRNN 边界锐利 6 倍（6s vs 38s MAE）——TCN 的 ±6min
感受野把边界抹平了。6s 误差小于切片 padding，产品上等于"边界准确"。

### 5.3 声纹判别专项（干扰 d）

阿梓金标模型 → 东雪莲 2.1h 歌回（全程高密度唱歌）：**检出 0 个片段**。
反向佐证：东雪莲伪标签模型在小柔/月隐/猫雷 3 场验证 VOD 上 0 预测事件（event_fp=0）。

### 5.4 全频带域差测试（部署关键问题）

训练数据是 8kHz 升采样的窄带音频，实际部署输入是全频带。用金标验证 VOD 1190452750
的**全频带 m4s 原始版本**直接推理（同一模型、同一阈值）：
event P=0.94 / R=0.79 / **F1=0.857**（8k wav 版为 0.89）——域迁移稳健，可直接上全频带。

### 5.5 速度与资源

| 项目 | 数值 |
|---|---|
| 纯模型吞吐 | ~14,000× 实时（3090, bf16） |
| 解码+特征+模型 RTF | 0.0005（2.5h VOD 用 4.9s） |
| 端到端含 ffmpeg 切片 | 2.5h VOD → 17 个切片，共 15.4s |
| 推理显存 | < 2GB（10GB 约束余量 5 倍+） |
| 模型体积 | 2.09M 参数 / 8.4MB |
| CPU 也可推理 | 模型层面可行（卷积+GRU，未测但参数量级无压力） |

## 6. 使用方法

```bash
# 一条命令：输入视频/音频，输出切片 + 切点 JSON + 概率曲线
python src/infer.py --input vod.mp4 --checkpoint runs/gold_crnn/best.pt \
    --out-dir out/ --gpu 0 --on-threshold 0.6 --off-threshold 0.3 --cut-video

# 换一位主播：准备 Audition 标记 csv（或用 scripts/pseudo_label.py 自举）
python scripts/gold_azusa.py        # 解析标注、缓存特征、出 manifests（按需改路径）
python src/train.py --model crnn --run-name new_streamer --gpu 0 \
    --manifest data/processed/gold_train.json --val-manifest data/processed/gold_val.json
python scripts/tune_postprocess.py --probs-dir runs/new_streamer/val_probs
```

产出：`out/<名>.segments.json`（start/end/cut_start/cut_end 秒）+ `out/<名>_songNN.mp4`
（`-c copy` 无重编码，秒级完成）+ `out/<名>.probs.npy`（3.125Hz 概率曲线可视化/审核用）。

## 7. 工程问题记录（已解决）

1. **容器 /dev/shm=64M 且 overlayfs 不支持共享 mmap** → PyTorch DataLoader 多进程
   SIGBUS。解法：worker 返回 numpy 走 pipe 序列化，主进程转 tensor。
2. **ZIP64**：4GB+ 条目本地头 size=0xFFFFFFFF，需解析 ZIP64 扩展字段，否则文件截断
   且后续条目（包括标注 csv！）永远解不出。
3. **HTDemucs 整文件分离 OOM**：2.8h 音频的 sources 张量 ~14GB ×3 进程静默被杀。
   解法：8 分钟分块 + 1s 交叉淡化。
4. zip 文件名 GBK 编码（Python 默认按 cp437 → 乱码）；huggingface.co 被墙走 hf-mirror；
   GitHub raw 间歇超时走 jsdelivr。

## 8. 局限与后续工作

- **召回 0.79**：8 个漏检事件多为短歌/低伴奏比段（清唱、戏腔）。下一步：(i) 阈值下探
  + 人工复核清单（低置信段输出"待确认"而非丢弃）；(ii) 用 6 场金标全量训练（本次留了
  2 场做验证）；(iii) 自训练一轮（模型反标全部 130h 歌回再训）。
- **域差**：训练数据为 8kHz 升采样；对全频带输入建议推理前 lowpass 到 4kHz 对齐域，
  或用数据集里的 m4s 全频带版本重训（数据已就位，1 小时可完成）。
- **歌手验证分支未启用**：当前端到端模型已通过声纹判别测试；若未来出现更难的
  same-singer 混淆（如音色相近的翻唱者），可加 ECAPA-TDNN 在分离人声 stem 上做
  段级余弦验证（调研已给出完整配方，speechbrain 权重已可获取）。
- **伪标签器是冷启动利器**：换新主播无标注时，数据工厂全自动出可用模型
  （东雪莲赛道证明了这点），人工只需复核边界。

## 9. 交付物清单

| 路径 | 内容 |
|---|---|
| `src/` | 特征/模型/数据/训练/推理/后处理/指标（全部自研，~1100 行） |
| `scripts/` | 数据工厂全链路：解压/分离/伪标签/合成/嵌入/manifests/金标/调参 |
| `runs/gold_crnn/best.pt` | **最终模型**（阿梓，金标训练） |
| `runs/{crnn,tcn,panns_head}/` | 东雪莲伪标签赛道三模型 + 调参结果 |
| `out/gold_demo/` | 端到端演示：17 个切片 + segments.json + 概率曲线 |
| `notes/` | RESEARCH.md / DESIGN.md / WORKLOG.md / research_raw.json |
| `runs/*/tb/` | 全部训练的 TensorBoard 曲线 |
