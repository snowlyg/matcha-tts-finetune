# 端侧推理约束（决定为什么只能微调 Matcha）

端侧走 **sherpa-onnx 离线推理**，硬约束：

- **sherpa-onnx 只支持三族推理 kernel：VITS / Matcha / Kokoro**。不属三族的模型
  （GPT-SoVITS、CosyVoice2、F5-TTS、ChatTTS、MegaTTS3、Pocket…）想用必须自己写 C++ kernel（周级），
  且这些模型本身要 GPU。
- **设备是 32 位 ARM（armeabi-v7a softfp）/ 部分 arm64，无 GPU**；RTF 预算 ≤2。
  自回归/flow 类模型端侧 RTF 普遍 10–40，直接出局。
- **端侧离线，不用服务端 TTS**——所以"云端跑大模型"不是端侧方案。

## 各模型端侧可行性

| 模型 | 授权 | 可微调 | 端侧可跑 | 结论 |
|---|---|---|---|---|
| Matcha zh-en（基座） | Apache | ✓（icefall） | ✓ RTF~1.3 | **本仓路线** |
| Kokoro multi-lang | Apache | 弱 | ✓ 但 310MB | 音色已被否 |
| GPT-SoVITS | MIT | ✓ 1 分钟 | ✗ 需 GPU | 只能当蒸馏老师 |
| CosyVoice2-0.5B | Apache(代码/权重) | ✓ 但端侧跑不了 | ✗ 需 GPU、非三族 | **当蒸馏老师**（zero-shot 产语料） |
| Pocket zh-en | CC-BY-NC | ✗ 仅 ONNX | ✗ | 出局 |

## 微调产物怎么回端侧
- 只换 `model-steps-3.onnx`；`vocos-16khz-univ.onnx`、`tokens.txt`、`lexicon.txt`、`espeak-ng-data`、
  `*-zh.fst` 都不动（前提：词表不变）。
- 换模型文件后设备要清 `filesDir/matcha-icefall-zh-en/` 才会重新下发（否则用旧缓存）。
