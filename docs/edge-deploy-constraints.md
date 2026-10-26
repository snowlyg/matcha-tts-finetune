# 端侧推理约束（决定为什么只能微调 Matcha）

端侧走 **sherpa-onnx 离线推理**，硬约束：

- **sherpa-onnx 的 TTS 推理 kernel 是逐族实现的**，本仓长期记录的"只有三族（VITS / Matcha /
  Kokoro）"**已过时**：上游自 **v1.12.11（`Add Zipvoice (#2487)`）** 起多了 **ZipVoice** 一族，
  后续 `v1.12.15` 加 C/Go API、`v1.12.19` 修 ZipVoice 英文分词、`v2.0.0` 修 ZipVoice 短文本堆溢出。
  仍不属任何一族的模型（GPT-SoVITS、CosyVoice2、F5-TTS、ChatTTS、MegaTTS3、Pocket…）想用必须自己
  写 C++ kernel（周级），且这些模型本身要 GPU。
- **设备是 32 位 ARM（armeabi-v7a softfp）/ 部分 arm64，无 GPU**；RTF 预算 ≤2。
  自回归/flow 类模型端侧 RTF 普遍 10–40，直接出局。
- **端侧离线，不用服务端 TTS**——所以"云端跑大模型"不是端侧方案。

## ZipVoice 换基座：本机实测已判死（2026-10-26）

同一台 Mac、同样 4 线程、同一批 5 句、各 10 次（ZipVoice 用 `num_steps=4`，已是对它有利的取值）：

| 引擎 | 模型 | RTF | 每句均耗时 |
|---|---|---|---|
| V1 Matcha（现网） | 73MB | **0.04** | 87 ms |
| ZipVoice | fp32 456MB（int8 约 104MB） | **1.33** | 2661 ms |

**同机耗时比 36.8×**。端侧 V1 实测 RTF≈1.3、预算 ≤2 → 就算 int8 把 ZipVoice 加速 3–5 倍、
再假设 ARM 的相对表现不打折，仍差一个数量级，**不可能进预算**。跨架构线性外推不精确，但 37 倍的
差距不是外推误差能填的。另有两条独立否决项：① ZipVoice 是**另一套 config + 要传参考音频**
（`OfflineTtsZipvoiceModelConfig` + `GenerationConfig.reference_audio`）→ **必须改 App 并重编译**，
"只换 `model-steps-3.onnx`"这条产品前提不成立；② 它是 zero-shot 克隆，音色不是现网那个。

**端侧运行时已取证（2026-10-26，真机）**：`libsherpa-onnx-c-api.so` → **sherpa-onnx 1.12.18**；
生产模型目录带 `lexicon.txt`(68,037 行，47,113 行是多音节短语)、`tokens.txt`(与训练词表 md5 相同)、
`number-zh.fst / date-zh.fst / phone-zh.fst`、`espeak-ng-data/`，**没有 `dict/`**。
两条含义：① 短语级发音覆盖**可以下发**（≥v1.12.10 支持短语最长匹配，≥v1.12.16 支持多份词典）；
② v1.12.15 起上游已把 jieba 从 matcha 前端移除、改用对 `lexicon.txt` 的最长匹配，
所以**本仓那套 jieba+pypinyin 复刻不等于设备前端**——取证细节与影响见 `docs/device-runtime-check.md`。

## 各模型端侧可行性

| 模型 | 授权 | 可微调 | 端侧可跑 | 结论 |
|---|---|---|---|---|
| Matcha zh-en（基座） | Apache | ✓（icefall） | ✓ RTF~1.3 | **本仓路线** |
| ZipVoice（换基座候选） | Apache | 未试 | ✗ 本机就比 V1 慢 37× | **判死**（见上节），且要改 App |
| Kokoro multi-lang | Apache | 弱 | ✓ 但 310MB | 音色已被否 |
| GPT-SoVITS | MIT | ✓ 1 分钟 | ✗ 需 GPU | 只能当蒸馏老师 |
| CosyVoice2-0.5B | Apache(代码/权重) | ✓ 但端侧跑不了 | ✗ 需 GPU、非三族 | **当蒸馏老师**（zero-shot 产语料） |
| Pocket zh-en | CC-BY-NC | ✗ 仅 ONNX | ✗ | 出局 |

## 微调产物怎么回端侧
- 只换 `model-steps-3.onnx`；`vocos-16khz-univ.onnx`、`tokens.txt`、`lexicon.txt`、`espeak-ng-data`、
  `*-zh.fst` 都不动（前提：词表不变）。
- 换模型文件后**设备侧有缓存**：必须清掉应用沙箱里那份模型目录才会重新下发（否则跑的还是旧模型）。
  开源鸿蒙上该目录的确切路径**要先在真机上探出来**（见 `docs/device-runtime-check.md`），
  不要照 Android 的 `filesDir` 猜。
