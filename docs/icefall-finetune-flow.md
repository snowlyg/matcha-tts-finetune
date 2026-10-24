# icefall Matcha 微调流程（GPU 机 / 云端）

目标：以作者 V1 pt 热启动，用自录医院话术微调，降低中文小数/轻声音节错率；
词表、声码器、端侧配置不变，只替换导出的 `model-steps-3.onnx`。

## 0. 环境
- Linux + NVIDIA GPU（24GB 最稳；12GB 可跑但要减 batch）。云可用 AutoDL。
- `git clone https://github.com/k2-fsa/icefall`；装 `requirements-tts.txt` +
  `piper_phonemize`（`pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html`，
  国内走 gh-proxy 下 wheel、保持规范 wheel 文件名再装）。
- 本仓 `scripts/` 用同一个 venv（pypinyin/onnxruntime/soundfile/kaldi_native_fbank/lhotse/sherpa-onnx）。

## 1. 语料 → 原始 cuts
```
python scripts/prepare_data.py --corpus <录音目录> --out data
# 产出 data/fbank/baker_zh_cuts_{train,valid,test}.jsonl.gz（带 supervisions[0].text）
```
没有真人录音时的替代：`scripts/gen_teacher_corpus.py` 批量合成 + SenseVoice 回读门控
（拒绝采样，只收念对的）。先用 `--backend v1 --limit 30` 自采样练手——实测（2026-10-24，
60 句录制稿）：首试通过 ~70%，但有 8 句重试 8 次全败，全是**系统性丢音节**
（患者李四→患者李、安排→安在），自采样修不了这类错 → 主路必须走老师模型。
主路是 `--backend cosyvoice`（CosyVoice2 zero-shot，`--ref-wav/--ref-text` 用 V1 现网
"温柔女声"片段定音色）——先小样本对照 ruler 基线验证老师错率 <5% 再放大产语料。
字母行（ICU 等）ASR 假错多，门控放宽、需人耳抽听。

## 2. 词表（钉死作者那份 + 补 `_ 0`）
```
python scripts/make_train_tokens.py --vocab models/vocab_tts.txt --out "$TTS/data/tokens.txt"
# = 作者 2190 token（id 1..2190 一位不动）前面补一行 `_ 0`，仅做 padding；
# 推理从不喂 `_`（V1 无 add_blank/BOS，见 author-model-facts.md）。不要跑 generate_tokens.py。
```
⚠ 训练必须关 intersperse（V1 推理不插 blank，导出物才能回落现网）：
把 `matcha/train.py` prepare_input 里 `intersperse_blank=True` 改为 `False`。

## 3. 前端（写 c.tokens）
```
cd egs/baker_zh/TTS
python local/convert_text_to_tokens.py \
  --in-file  <repo>/data/fbank/baker_zh_cuts_train.jsonl.gz \
  --out-file <repo>/data/fbank/baker_zh_cuts_train.tok.jsonl.gz
# valid/test 同理
```
（jieba + lazy_pinyin(TONE3, tone_sandhi=True) + piper_phonemize，与端侧推理前端同源；
自录转写要先规整：数字转汉字、字母拆读，参考 data/record_script.tsv。）

## 4. 特征 + cmvn（V1 是 16k）
- 复制 `local/compute_fbank_baker_zh.py`，把 `sampling_rate=22050`、`.resample(22050)`
  改为 **16000**（f_max 保持 8000=16k 奈奎斯特；n_fft/hop/win/mels 不动：1024/256/1024/80）。
- `local/compute_fbank_statistics.py` 出自录语料的 cmvn → `data/fbank/cmvn.json`。
- 作者 ckpt 的 mel buffer (-2.95/5.45) 与公共配置对不上（详见 author-model-facts），
  统一改用**自算 cmvn 并覆写 buffer**（下一步脚本已支持）。

## 5. 热启动
```
# 注意：基座必须是 pytorch_model_sherpa.bin（与现网 steps-3 ONNX 权重一致；
# pytorch_model.bin 是另一份，303/305 张量不同）
python <repo>/scripts/make_warmstart_ckpt.py \
  --bin models/zh-en/pytorch_model_sherpa.bin \
  --tokens "$TTS/data/tokens.txt" \
  --out matcha/exp-1/epoch-1.pt \
  --mel-mean <cmvn均值> --mel-std <cmvn标准差>
# 脚本会剥 model. 前缀、把 emb 2190→2191 行（ids 与作者完全一致）
python matcha/train.py --exp-dir matcha/exp-1 --start-epoch 2 \
  --num-epochs 30 --max-duration 1200 --bucketing-sampler 1 --num-workers 4 --world-size 1
# 12GB 显存：--max-duration 降到 200–400
```
首个 step 前人工确认 load 无 missing/unexpected keys（icefall load 默认 strict=False，
不报错不代表键名对上；可临时 strict=True 验证一次）。

## 6. 导出 ONNX
```
python matcha/export_onnx.py --exp-dir matcha/exp-1 --epoch <N> \
  --tokens data/tokens.txt --cmvn data/fbank/cmvn.json
# 产出 model-steps-2..6.onnx；取 steps-3 替换端侧那份
```
需 `torch>=2.2.0`（否则 SDPA 导出报错）。
⚠ export_onnx.py 元数据默认 `sample_rate: 22050`，须改为 **16000**（现网 V1 即 16000，
端侧按它定时长/采样）；`use_eos_bos` 保持 0、不要引入 `add_blank` 键
（sherpa-onnx 靠这些键决定插 blank/BOS）。导出后与 V1 元数据逐键 diff 一遍。

## 7. 验收
- 把导出的 `model-steps-3.onnx` 放进 `models/zh-en/`，跑 `scripts/ruler.py --n 100`，
  对比微调前后错率。达标线：中文各条 <5%、21床/字母不回归、RTF 不明显变差。
- 通过后再走宿主项目的 HAR 出包/装机流程（端侧替换 rawfile 模型 + 清 filesDir 缓存）。
