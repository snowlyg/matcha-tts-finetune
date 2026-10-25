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

本仓 PC 复现尺子（2026-10-24，steps-3，noise/length=1.0，n=100）：
体温 45%、为您自动接听 59%、21床 3%、A一零床 5%、ICU 17%（含 ASR 把字母名拼成
单词的假错），总体 129/500=25.8%。与 n=20 基线同型（扩散采样方差大，趋势一致），
`out/ruler_baseline.log` 为证；**微调验收以这套 PC 尺子同参数复测对比**。

## G2P（文本→token）
- 作者 V1 测试脚本 `test_onnx_20251203_zh_en.py` 内含前端：中文 `pypinyin`、英文
  `piper_phonemize` + `convert_to_gruut_en_us_strict`。
- **训练前端其实开源在 icefall**：`egs/baker_zh/TTS/local/convert_text_to_tokens.py`
  = **jieba 分词 → `lazy_pinyin(style=TONE3, tone_sandhi=True)` → `piper_phonemize`**，
  写进 cut 的 `c.tokens`。`tone_sandhi=True` 只处理 一/不、不做三声连读（与作者一致）。
- 本仓 `scripts/g2p.py` 对齐该逻辑（残差：未做 jieba 分词的个别多音字）。
- 实测（2026-10-24）：作者测试脚本**自带前端**（整串 `pinyin()`、无 tone_sandhi、
  无 jieba、标点未映射）**复现不了它自己注释里的硬编码 id**（18 处差异，全标点变空格）；
  本仓 g2p（tone_sandhi、无 jieba）只差 4 处（多音字/变调），是三个候选前端里最近的。
  硬编码 ids 更像出自设备侧运行时（带分词的中文前端）。训练侧以 icefall
  `convert_text_to_tokens.py`（jieba+tone_sandhi）为准，两端同源要在验收时回归确认。

## 特殊 token / 词表对齐（2026-10-24 已核实，证据见下）
- 作者 `tokens.txt`：id **1..2190**（=vocab_tts.txt 行号+1），无 `_`；空格在本词表里就是 id1。
- **无 BOS/EOS**：作者硬编码 id 序列首项 414 = `dang4`，正是「当夜幕降临」的「当」；
  两个作者测试脚本均不前置特殊 token；sherpa-onnx `offline-tts-matcha-impl.h` 注释
  明确 "the zh-en model doesn't use them"（`ApplyBosEos` 对 zh-en 是直通）。
  ONNX 元数据 `use_eos_bos:1` 对 zh-en 运行时不生效，勿据此加 BOS。
  反向同理：本仓 `matcha/export_onnx.py` 写的是 `use_eos_bos: 0`，**不要"对齐 V1"改成 1**
  （我们的训练约定确实没有 BOS/EOS，0 才是诚实声明；1 在 zh-en 下惰性，改了也没后果）。

### V1 `model-steps-3.onnx` 元数据全量（2026-10-25 从 `metadata_props` 直接 dump，共 12 键）
导出后与这张表逐键 diff；差异只允许出现在"我们有意要改"的地方。
```
language        = chinese English        model_author  = dengcunqin
model_type      = matcha-tts             version       = 1
n_speakers      = 1                      num_ode_steps = 3
pad_id          = 0                      sample_rate   = 16000
use_eos_bos     = 1                      voice         = zh en-us
demo_url        = https://www.tulingyun.com/tts.html
modelscope_url  = https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010
```
- `pad_id=0` 与我们 `_ 0` 词表一致；**没有 `add_blank` 键**（sherpa-onnx 靠它决定插 blank）。
- 现网这份的 emb 是 `model.encoder.emb.weight (2190,192)`，输入 `x[N,L] int32` /
  `x_length[N]` / `noise_scale[1]` / `length_scale[1]`，输出 `mel[N,80,L]`。
  `export_onnx.py:190` 的 `sample_rate` 默认 **22050**，必须 sed 成 16000（第 196 行 `use_eos_bos` 已是 0）。
- **训练从不插 blank**：ONNX 元数据无 `add_blank`，端侧 ids 直进直出且效果正常
  → 微调必须关掉 icefall 的 `intersperse_blank=True`（matcha/train.py prepare_input），
  否则导出物与现网推理语义不符。
- `pytorch_model.bin` ≠ `pytorch_model_sherpa.bin`：305 个张量里 303 个不同。
  **现网 model-steps-3.onnx 的 embedding 与 `pytorch_model_sherpa.bin` 逐元素相同**
  → 热启动基座用 sherpa 那份，才是可回落现网的权重。
- 作者 emb 仅 **2190 行**（`model.encoder.emb.weight (2190,192)`），id 1..2190 意味着
  id2190（`<|unused_2190|>`）是**潜伏越界**，因其从不被喂入而存活；id0 是未用死行。
- **补法（scripts/make_train_tokens.py + make_warmstart_ckpt.py）**：训练 tokens.txt
  在作者词表前补 `_ 0`（只做 padding，被 x_mask 遮蔽，推理从不喂入），
  作者 2190 个 id **一位不动**；ckpt 剥 `model.` 前缀、emb 扩 1 行（新行 normal init，
  对应从不喂入的 id2190）。导出的 ONNX 因此可直接回落到现网运行时。
  绝不跑 `local/generate_tokens.py`。

## V1 特征域（部分核实，仍需问作者）
- 采样率 **16k 确认**：ONNX 元数据 `sample_rate:16000`（icefall 默认导出写 22050，
  是作者改过）、vocos-16khz-univ、端侧音频节奏正常（若按 baker 86fps mel 栅格会明显变速）。
- ckpt 内 buffer：`mel_mean=-2.9522, mel_std=5.4468`（标量）。用公共
  MatchaFbank(n_fft1024,hop256,win1024,mels80,fmin0,fmax8000/7600,center=False)
  对合成音频重算 mel 对不上该统计（受响度/语料动态范围影响，无法单独定论 fmax 与
  归一化细节）→ **微调自算 cmvn，并用 make_warmstart_ckpt.py --mel-mean/--mel-std
  覆写 buffer**，保证训练归一化域一致；导出 --cmvn 用同一份。
- 待作者确认：其 fork 的 fbank 精确参数与语料响度规范。

## 其他候选（都出局）
- **Pocket zh-en**（`dengcunqin/pocket-tts-zh-en-pinyin`）：CC-BY-NC（non-commercial）、只发 ONNX（不可微调）、
  非 sherpa 三族、24k、需 Python runtime → 不能上端侧、不能当微调基座。
- **GPT-SoVITS / CosyVoice2 / F5-TTS / ChatTTS 等**：MIT/各异、少样本克隆强，但都是 GPT/flow 自回归、
  需 GPU、非 sherpa 三族 → 端侧跑不了。唯一可用法：当"蒸馏老师"（先验证它对目标话术确实更准）。

## 待核实的技术问题
- 训练 fbank 精确参数与语料响度规范（V1 的 mel buffer 与公共配置对不上，见上节）。
- `pytorch_model.bin` 与 `pytorch_model_sherpa.bin` 的差别（哪份更新）。
- 已自证不再需要问：BOS（无）、是否 16k（是）。
