# AutoSlicer v2 — 歌回直播自动切片

**给它一个 B 站链接,它还你一首一首切好的歌。**

主播的歌回动辄三四个小时,想把每首歌单独存下来,要么手动拖进度条找切点,要么等切片
man 慢慢更新。AutoSlicer 用一个 2.1M 参数的小模型自动完成这件事:识别出**主播本人
在唱歌**的每一段,直接用 ffmpeg 无损切出独立视频——3 小时的录播,全程不到 1 分钟。

```bash
python src/infer.py --input "https://www.bilibili.com/video/BVxxxxxxxxxx" \
    --checkpoint best.pt --out-dir out/ --cut-video
```

```
out/
├── xxx_song01.mp4      ← 第 1 首歌(含前后 5 秒余量)
├── xxx_song02.mp4      ← 第 2 首歌
├── ...
├── xxx.segments.json   ← 每段的起止时间、耗时统计
└── xxx.probs.npy       ← 逐帧"正在唱歌"概率曲线(可视化/二次开发用)
```

## 它聪明在哪

普通的"唱歌检测"模型分不清下面这些场景,AutoSlicer 是按这些场景专门设计训练的:

| 场景 | 行为 |
|---|---|
| 主播唱歌 | ✂️ 切出来 |
| 主播放别人的歌、放原唱参考 | 🚫 跳过——它认的是**这个人的声音**,不是"有人在唱" |
| 间奏、纯伴奏、调音试麦 | 🚫 不会被伴奏骗到 |
| 唱了一分钟 → 放了一段原唱对比 → 接着唱 | ✂️ 合并成**一个**完整切片,不会切碎 |
| 唱完聊天五分钟再唱下一首 | ✂️ 切成两个独立切片 |
| 闲聊、读弹幕、打游戏 | 🚫 跳过 |

实测把"阿梓检测器"喂给另一位主播 2.1 小时全程唱歌的歌回:**0 个误切**。

## 性能

- **准确率**:独立测试集(从未参与训练调参的 2 场录播)事件级 F1 **0.90**——45 首歌
  命中 40 首、误切 4 段;3 折交叉验证 F1 0.93 ± 0.03
- **速度**:2.5 小时 VOD 从解码到切完 **15 秒**(不含下载);模型推理是实时的 14000 倍
- **硬件**:显存 <2GB,任何近十年的 N 卡都跑得动;没有 GPU 加 `--device cpu` 也只要几分钟
- **体积**:模型 8MB,单文件

## 安装

```bash
git clone https://github.com/anAtheist987/AutoSlicer-v2.git && cd AutoSlicer-v2
pip install torch torchaudio numpy scipy requests   # 推理只需要这 5 个
# ffmpeg 需要在 PATH 里:apt install ffmpeg / brew install ffmpeg

# 下载模型(阿梓声线版,Releases 页也可手动下)
wget https://github.com/anAtheist987/AutoSlicer-v2/releases/download/v2.1.0/best.pt
```

训练自己的模型才需要完整依赖:`pip install -r requirements.txt`。

## 使用

### 1. B 站链接直接切(最常用)

```bash
python src/infer.py --input "https://www.bilibili.com/video/BVxxxxxxxxxx" \
    --checkpoint best.pt --out-dir out/ --cut-video
```

BV 号、av 号、`b23.tv` 短链都认,会自动模拟浏览器下载后切片:

```bash
python src/infer.py --input BVxxxxxxxxxx --checkpoint best.pt --cut-video
python src/infer.py --input "https://b23.tv/xxxxxx" --checkpoint best.pt --cut-video
```

- 多 P 录播用 `--page 2` 选分 P
- 不登录能拿到 480p 视频 + 192kbps 音频(识别完全够用);想要高清切片,从浏览器
  开发者工具里复制自己的 Cookie:`--cookie "SESSDATA=..."`
- 不加 `--cut-video` 则只下音频做检测,几秒钟出 `segments.json` 切点表,适合先
  预览再决定切不切

### 2. 本地文件

录播姬、BBDown 等工具下好的文件直接喂,视频音频格式都行(mp4/flv/mkv/m4a/mp3/wav...):

```bash
python src/infer.py --input 录播.flv --checkpoint best.pt --out-dir out/ --cut-video
```

### 3. 只用下载器

```bash
python src/bilibili.py "https://www.bilibili.com/video/BVxxxxxxxxxx" --out-dir downloads
python src/bilibili.py BVxxxxxxxxxx --video        # 连视频一起下并合流成 mp4
```

支持断点续传和备用 CDN 自动切换。

### 4. 读懂输出

`segments.json` 里每段长这样:

```json
{
  "start": 1234.5,        // 检测到开唱的时刻(秒)
  "end": 1480.2,          // 检测到唱完的时刻
  "cut_start": 1229.5,    // 实际切片起点(自动加了 5 秒余量)
  "cut_end": 1485.2
}
```

切片用 `ffmpeg -c copy` 流复制,**不重编码**:画质无损、切 17 首歌只要几秒。

### 5. 调节松紧

| 想要 | 加参数 |
|---|---|
| 少漏歌(宁可多切) | `--on-threshold 0.4` |
| 少误切(宁可漏) | `--on-threshold 0.7` |

进阶旋钮在 `src/postprocess.py` 的 `PostProcessConfig` 里,都有注释:多近的两首歌合并
成一个切片(`merge_gap_s`,默认 90 秒,这就是"唱歌→放原唱→接着唱算一首"的来源)、
最短算一首歌的时长(`min_song_s`)、切片前后余量(`pad_*_s`)等。

## 训练你自己主播的模型

仓库自带的模型认的是[阿梓](https://space.bilibili.com/7706705)的声线。换主播需要重训,
两条路:

### 路线 A:有人工标注(效果最好)

用 Adobe Audition / 剪映等工具在几场歌回上把每首歌标出来,导出 tab 分隔 csv
(Name/Start/Duration 三列)。4-6 场标注就够达到 F1 0.9 水平:

```bash
# 1. 音频转 16kHz 全频带 log-mel 缓存
python scripts/cache_fullband.py
# 2. 由标注生成逐帧标签 + 训练/验证清单(参考 scripts/gold_azusa.py 改主播名)
python scripts/build_v2_labels.py
# 3. 训练(单卡 ~45 分钟,8.9GB 显存;batch 减半则 ~4.5GB)
python src/train.py --model crnn --run-name my_streamer --gpu 0 \
    --classes 4 --bandlimit-prob 0.4 \
    --manifest data/processed/v2_train.json --val-manifest data/processed/v2_val.json \
    --steps 8000 --val-every 1000
# 4. 在验证集上自动选后处理阈值
python scripts/eval_dump.py --checkpoint runs/my_streamer/best.pt \
    --val-manifest data/processed/v2_val.json --out-dir runs/my_streamer/val_probs
python scripts/tune_postprocess.py --probs-dir runs/my_streamer/val_probs
```

### 路线 B:一行标注都没有(自监督冷启动)

数据工厂全自动:HTDemucs 把歌回分离成人声/伴奏 → 用"人声能量 + 音高稳定度"打
伪标签(唱歌时音高有持续稳定段,说话没有)→ 拿其他主播的歌回当"他人唱歌"难负例 →
stem 重混合成"伴奏配错人声"等干扰样本:

```bash
python scripts/separate_all.py --gpu 0      # 人声/伴奏分离(最耗时的一步)
python scripts/pseudo_label.py              # 伪标签
python scripts/make_synthetic.py --target 歌回【你的主播】
python scripts/build_manifests.py --target 歌回【你的主播】
python src/train.py ...                     # 同上
```

我们用这条路线在零标注下做到了伪标签验证 F1 0.98。两条路线可以接力:先 B 后 A。

## 常见问题

**没有 N 卡能用吗?** 能。`--device cpu`,2.5 小时 VOD 大约 3-5 分钟出结果。

**为什么有一首歌没切出来 / 多切了一段?** 看 `probs.npy` 概率曲线就知道模型当时的
"心理活动";通常调 `--on-threshold` 就能解决。清唱(无伴奏)和 1 分钟以内的短歌
是已知弱项。

**切点准吗?** 平均误差十秒上下,且默认前后各放 5 秒余量,一般不掐头去尾。要精修
可以拿 `segments.json` 导入剪辑软件微调。

**直播间直接切吗?** 目前面向录播(VOD)。模型本身是流式友好的(因果卷积 + GRU),
实时版在路线图上。

**模型多大,要什么环境?** 8MB / PyTorch ≥ 2.0 / Python ≥ 3.10。

## 技术细节

架构、四类输出头设计、自监督数据工厂、交叉验证协议、全部实验数字,见
**[reports/FINAL_REPORT.md](reports/FINAL_REPORT.md)**;SOTA 调研在
[notes/RESEARCH.md](notes/RESEARCH.md)。一句话版本:16kHz 全频带 log-mel → 4 层
CNN(16× 时间池化)→ BiGRU → 4 类 powerset 帧分类(无人声 / **本人唱** / 他人唱 /
纯伴奏)→ 中值平滑 + 滞回双阈值 + 间隙合并 → ffmpeg。

| 路径 | 内容 |
|---|---|
| `src/` | 模型 / 特征 / 训练 / 推理 / 后处理 / B站下载 |
| `scripts/` | 数据工厂(解压/分离/伪标签/合成/标签/调参/评估) |
| `notes/` `reports/` | 调研、设计、工作日志、最终报告 |

## 许可证与声明

代码 [MIT](LICENSE)。训练数据(主播 VOD)不随仓库分发。B 站下载模块仅供个人学习
研究:请遵守 bilibili 服务条款,尊重主播对其内容的权利,**切片二次发布前请确认获得
主播授权**。前代项目:[AutoSlicer v1](https://github.com/anAtheist987/AutoSlicer)(2021)。
