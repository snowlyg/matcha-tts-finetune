# 模型与作者侧事实（2026-10 取证）

## 现网模型
- `matcha-icefall-zh-en`（中英混读 Matcha + vocos-16k），源自 ModelScope
  `dengcunqin/matcha_tts_zh_en_20251010`（V1，「温柔女声」，**Apache-2.0**）。
- 微调基座用 V1 的 `pytorch_model.bin`（74,490,446B）。base 仓 `matcha_tts_zh_en` 的
  ONNX 是 139MB 档、V1 是 75MB 档，**不是同一个导出物**，别混。
- 词表：V1 的 `vocab_tts.txt` 与设备端 `tokens.txt` 逐行相同，2190 条、无重复，映射 `idx+1`。

## 内容错率（实测，n=20，steps-3，noise/length=1.0）
| 话术 | 错率 | 性质 |
|---|---|---|
| 体温三十六点五度 | 30% | 真实模型错（小数） |
| 为您自动接听 | 70% | 真实模型错（轻声/叠字） |
| 二十一床正在呼叫 | 0% | 正确 |
| A 一零床 / ICU | ≈0 | 字母读对（ASR 把字母名拼成单词属假错） |

调参（noise 0.2–1.3、speed、ODE steps 3→6、fst、数字走汉字）全部证否——是训练分布问题，不是采样。

## G2P（文本→token）
- 作者 V1 测试脚本 `test_onnx_20251203_zh_en.py` 内含前端：中文 `pypinyin`、英文
  `piper_phonemize` + `convert_to_gruut_en_us_strict`。
- **训练前端其实开源在 icefall**：`egs/baker_zh/TTS/local/convert_text_to_tokens.py`
  = **jieba 分词 → `lazy_pinyin(style=TONE3, tone_sandhi=True)` → `piper_phonemize`**，
  写进 cut 的 `c.tokens`。`tone_sandhi=True` 只处理 一/不、不做三声连读（与作者一致）。
- 本仓 `scripts/g2p.py` 对齐该逻辑（残差：未做 jieba 分词的个别多音字）。

## ⚠ 特殊 token 待核（上 GPU 机第一件事）
- 作者 `tokens.txt`：id 从 **1** 起、**无 `_` pad、无 id0 blank**。
- icefall `matcha/tokenizer.py`：`pad=token2id["_"]`、blank 习惯 id0 → **直接套会 KeyError**。
- 训练词表必须钉死作者那份，**不要跑 `generate_tokens.py`**（那是 baker 纯中文词表，会换掉 vocab 破坏兼容）。

## 其他候选（都出局）
- **Pocket zh-en**（`dengcunqin/pocket-tts-zh-en-pinyin`）：CC-BY-NC（禁商用）、只发 ONNX（不可微调）、
  非 sherpa 三族、24k、需 Python runtime → 不能上端侧、不能当微调基座。
- **GPT-SoVITS / CosyVoice2 / F5-TTS / ChatTTS 等**：MIT/各异、少样本克隆强，但都是 GPT/flow 自回归、
  需 GPU、非 sherpa 三族 → 端侧跑不了。唯一可用法：当"蒸馏老师"（先验证它对医院话术确实更准）。

## 授权红线
- baker 数据集 **non-commercial**；医院是商用 → 训练语料必须自录/已清权，交付前过法务。
- 可商用替代中文数据：AISHELL-3、DiDiSpeech（均 Apache-2.0）；英文 LibriTTS/LJSpeech。

## 联系作者
`2735033883@qq.com` / 微信 `game_for_enjoy` / FunASR 群「找小邓」。待确认：是否要 BOS（首 id 414）、
训练采样率/fbank 配置（V1 是 16k，别照搬 baker 22.05k）、V2 何时发。
