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

## 2. 词表（钉死作者那份）
```
# 把作者 V1 的 tokens.txt 直接当 data/tokens.txt；不要跑 generate_tokens.py
cp models/zh-en/tokens.txt data/tokens.txt
```
⚠ 核对 pad/blank 与 icefall `tokenizer.py` 是否一致（作者无 `_`/id0），必要时小改 tokenizer。

## 3. 前端（写 c.tokens）
```
cd egs/baker_zh/TTS
python local/convert_text_to_tokens.py \
  --in-file  <repo>/data/fbank/baker_zh_cuts_train.jsonl.gz \
  --out-file <repo>/data/fbank/baker_zh_cuts_train.tok.jsonl.gz
# valid/test 同理
```

## 4. 特征 + cmvn
```
python local/compute_fbank_baker_zh.py ...   # 产出带 fbank 的 cuts
python local/compute_fbank_statistics.py ...  # data/fbank/cmvn.json
```
（具体参数以本机 icefall 版本 `--help` 为准；16k 采样率要对齐 V1，别用 baker 默认 22.05k。）

## 5. 热启动
icefall `train.py` 只从 `exp-dir/epoch-(start-epoch-1).pt` 续训：
```
# 把 V1 pytorch_model.bin 放成 exp-1/epoch-1.pt（可能需对齐 state_dict key），然后
python matcha/train.py --exp-dir matcha/exp-1 --start-epoch 2 \
  --num-epochs 30 --max-duration 1200 --bucketing-sampler 1 --num-workers 4 --world-size 1
# 12GB 显存：--max-duration 降到 200–400
```

## 6. 导出 ONNX
```
python matcha/export_onnx.py --exp-dir matcha/exp-1 --epoch <N> \
  --tokens data/tokens.txt --cmvn data/fbank/cmvn.json
# 产出 model-steps-2..6.onnx；取 steps-3 替换端侧那份
```
需 `torch>=2.2.0`（否则 SDPA 导出报错）。

## 7. 验收
- 把导出的 `model-steps-3.onnx` 放进 `models/zh-en/`，跑 `scripts/ruler.py --n 100`，
  对比微调前后错率。达标线：中文各条 <5%、21床/字母不回归、RTF 不明显变差。
- 通过后再走宿主项目的 HAR 出包/装机流程（端侧替换 rawfile 模型 + 清 filesDir 缓存）。
