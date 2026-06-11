# AutoSlicer v2 调研报告

> 来源：6 个并行调研 agent（SVD 文献 / 现代骨干 / 生产 VAD / 歌手验证 / 源分离 / 后处理），
> 原始结构化结果见 `notes/research_raw.json`（含全部 URL 引用）。2026-06-12 凌晨完成。

## 1. 任务回顾

在数小时直播 VOD 中检测「主播 A 本人在唱歌」，每首歌切成独立视频段。干扰：(a) 纯伴奏未唱
(b) 伴奏与唱不匹配 (c) 伴奏只是歌的一部分 (d) 放别人唱的同一首歌（对照跑调）。
约束：3090 一夜训完；<10GB GPU 推理；RTF < 1~2；准确率 ≥80%，目标 >90%。

## 2. 核心结论（跨方向共识）

1. **生产级系统形态完全收敛**：log-mel → 小型逐帧分类器（0.1M–4M 参数）→ 滑窗 overlap-add
   → 双阈值滞回 + 最短时长/间隙填充 → 段落。Silero VAD（~2MB，单核 32ms chunk <1ms）、
   pyannote 3.x、NVIDIA MarbleNet（91.5K 参数即可做 20ms 帧标注）、inaSpeechSegmenter 全是这个形状。
   **不要重新发明这个轮子。**
2. **输出头应为多类而非二分类**（pyannote 3.x powerset 思路，Interspeech 2023）：
   {A在唱, 其他人声演唱/放录音, 纯音乐无人声, 说话/无} 的 softmax。
   干扰 (a)-(c) 变成显式的「纯音乐」类，(d) 变成显式的「其他人声」类，
   而不是指望二分类模型隐式学会。解码时「A在唱」= argmax 为该类。
3. **纯 SVD 的天花板**：Jamendo 基准上 2015 年 CNN（Schlüter&Grill）即 89-92% 帧准确率，
   十年只推到 92-94%（标注噪声封顶）。我们的任务 = SVD × 歌手验证，难点在「是谁在唱」。
4. **源分离当离线数据工厂用，不进推理链路**：HTDemucs 2-stem（MIT，42M，3090 上 >10× RT，3-7GB）
   批量把歌回 VOD 拆成 人声/伴奏 stem，可程序化制造全部四类干扰的训练数据，特别是杀手级难负例：
   **A 的伴奏 stem + 原唱的人声 stem 混合 = 干扰 (d) 的完美合成样本**（ICASSP 2021 合成混音配方）。
   分离出的 A 人声 stem 还可用于歌手注册（enrollment）。
5. **干扰 (d) 没有任何 tagging/VAD 模型能单独解决** —— 需要歌手身份分支：
   ECAPA-TDNN（speechbrain，Apache-2.0，6M 参数）/CAM++ 在**分离后的人声 stem** 上取 3-5s 窗口嵌入，
   与 A 的歌声注册质心做余弦 + 段级聚合（中位数/top-k均值）。三个独立来源（JukeBox Interspeech 2020、
   SingFake、Deezer ISMIR 2024）证实：语音训练的声纹模型在歌声+伴奏上显著退化，
   **必须在分离 stem 上跑 + 用 A 的歌声（而非说话声）注册**。级联运行（只在候选唱歌区域上跑）省算力。
6. **长上下文不再是问题**：现代做法是 10s 级窗口的骨干 + 1Hz 级第二阶段序列模型补足分钟级上下文。
   4 小时 VOD @1Hz 仅 ~14.4k token —— 全注意力都算得起；膨胀 TCN（MS-TCN 风格，1-3M 参数，
   感受野 10+ 分钟，3090 上几分钟训完）是性价比之王。旧系统「需要数分钟卷积」的痛点已消解。
7. **预训练骨干现成可用**：fschmid56/PretrainedSED（MIT，自动下载权重）提供
   frame_mn10（3.83M，40ms 帧分辨率，PSDS1 距 transformer SOTA ~1 分）与
   ATST-F/BEATs strong（~90M，精度上限）。EfficientAT mn10（4.88M，比 PaSST 快 61×）。
   PANNs CNN14_16k（80M，Zenodo 直链）经典稳健。CED-Tiny/Mini（5.5M/9.6M）。
   注意 MERT 是 CC-BY-NC，避免。
8. **指标与调参**：不要用 frame-F1 选模型 —— 用事件级 F1（带 5-10s collar）+ 交并比类
   PSDS 标准（检测覆盖歌曲 ≥70% 且 ≥70% 在歌内）。后处理五个常数（on/off 阈值、
   min_duration_on/off、gap-fill）在 dev VOD 上网格调，是最便宜的精度来源
   （Cances et al. 2019：仅后处理即可大幅改变 event-F1；2024 SOTA 是 cSEBBs 变点法）。
9. **gap-fill 与干扰 (a)-(c) 的交互陷阱**：大间隙填充会把「唱歌前后播放的伴奏」也并进片段。
   缓解：间隙填充仅在两侧均为高置信 A 唱歌时进行，或用「伴奏延续」信号门控。
10. **可选加分项**：视频模态 —— ByteDance 直播 AV-VAD（arXiv:2010.14168）证明嘴部运动相关性
    可直接消解「放录音」歧义；切点可用音乐结构边界（allin1/自相似新颖度）精修。

## 3. 骨干候选对比

| 模型 | 参数 | 帧分辨率 | 权重 | 许可 | 评注 |
|---|---|---|---|---|---|
| MarbleNet 风格 1D-conv | 0.1-1M | 20ms | 自训 | - | 吞吐之王，能力有限 |
| PyanNet (pyannote seg) | 1.5M | 17ms | HF | MIT | 生产验证，SincNet+BiLSTM |
| 自建 Mel-CRNN | 2-4M | 320ms | 自训 | - | 基线，必然可控 |
| frame_mn10 (PretrainedSED) | 3.83M | 40ms | GitHub | MIT | AudioSet-strong 预训练帧级 SED，速度赛道首选 |
| EfficientAT mn10 | 4.88M | clip级 | GitHub | MIT | 需加帧头 |
| CED-Tiny/Mini | 5.5/9.6M | clip级 | HF | Apache | 蒸馏 transformer |
| ECAPA-TDNN | 6M | 嵌入 | speechbrain | Apache | 歌手验证分支 |
| PANNs CNN14_16k | 80M | 640ms | Zenodo✓已下载 | MIT | 冻结嵌入+轻头，稳健 |
| ATST-F/BEATs strong | ~90M | 40ms | GitHub | MIT | 精度上限赛道，3090 可微调 |
| HTDemucs 2-stem | 42M | - | pip | MIT | 离线数据工厂 |

## 4. 对本项目设计的直接启示

- 共享 mel 前端 + 多类帧头 + 强后处理是地板；歌手分支是过 (d) 的天花板钥匙。
- 数据侧杠杆 > 模型侧杠杆：分离-重混能一夜制造出覆盖全部四类干扰的、与目标主播
  声学信道完全匹配的训练分布。
- 二阶段（1Hz 序列模型）将歌曲级语义（间奏、整首歌结构）与帧级声学解耦，
  训练快、推理近零成本，正面替代旧系统「数分钟卷积感受野」的设计。
