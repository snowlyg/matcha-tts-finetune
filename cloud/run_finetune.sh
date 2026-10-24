#!/usr/bin/env bash
# 云端/GPU 机一键微调（骨架）。运行前填好下面的路径，并确保 icefall 已 clone、
# requirements-tts.txt + piper_phonemize 已装、自录语料已就位。
# 各 icefall 脚本的精确参数以本机版本 --help 为准；四大阻塞已核实，
# 结论与依据见 docs/author-model-facts.md、操作细节见 docs/icefall-finetune-flow.md。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ICEFALL="${ICEFALL:-$HOME/icefall}"
CORPUS="${CORPUS:-$REPO/data/corpus}"          # 含 wav/ 与 transcripts.tsv
TTS="$ICEFALL/egs/baker_zh/TTS"
PY="${PY:-python3}"

echo "[1] 语料 -> 原始 cuts"
"$PY" "$REPO/scripts/prepare_data.py" --corpus "$CORPUS" --out "$REPO/data"

echo "[2] 词表：作者 2190 token（id 1..2190 不动）+ 首行 _ 0 作 padding；绝不跑 generate_tokens.py"
mkdir -p "$TTS/data"
"$PY" "$REPO/scripts/make_train_tokens.py" --vocab "$REPO/models/vocab_tts.txt" \
  --out "$TTS/data/tokens.txt"

echo "[2b] 关掉 intersperse_blank（V1 推理不插 blank，否则导出物与现网语义不符）"
grep -q "intersperse_blank=True" "$TTS/matcha/train.py" && \
  sed -i 's/intersperse_blank=True/intersperse_blank=False/' "$TTS/matcha/train.py"
echo "    已确认: $(grep -n 'intersperse_blank=' "$TTS/matcha/train.py")"

echo "[3] 前端：jieba + lazy_pinyin(tone_sandhi) + piper -> c.tokens"
cd "$TTS"
for split in train valid test; do
  "$PY" local/convert_text_to_tokens.py \
    --in-file  "$REPO/data/fbank/baker_zh_cuts_${split}.jsonl.gz" \
    --out-file "$REPO/data/fbank/baker_zh_cuts_${split}.tok.jsonl.gz"
done

echo "[4] 特征 + cmvn：把 local/compute_fbank_baker_zh.py 的 22050 改 16000（V1=16k）后手动跑，"
echo "    再跑 local/compute_fbank_statistics.py 得 data/fbank/cmvn.json。"
echo "    （n_fft/hop/win/mels = 1024/256/1024/80、f_max 8000 保持；见 flow 文档第 4 步）"

echo "[5] 热启动：基座必须是 pytorch_model_sherpa.bin（与现网 steps-3 ONNX 权重一致）"
# 拿到 [4] 的 cmvn 后，补 --mel-mean/--mel-std 覆写 buffer（防止沿用作者 -2.95/5.45）：
"$PY" "$REPO/scripts/make_warmstart_ckpt.py" \
  --bin "$REPO/models/zh-en/pytorch_model_sherpa.bin" \
  --tokens "$TTS/data/tokens.txt" \
  --out matcha/exp-1/epoch-1.pt
# 首次建议临时 strict=True 验证 load 无 missing/unexpected keys，然后：
#   python matcha/train.py --exp-dir matcha/exp-1 --start-epoch 2 \
#     --num-epochs 30 --max-duration 1200 --bucketing-sampler 1 --num-workers 4 --world-size 1
#   （12GB 显存把 --max-duration 降到 200-400）

echo "[6] 导出：python matcha/export_onnx.py --exp-dir matcha/exp-1 --epoch <N> \\"
echo "      --tokens data/tokens.txt --cmvn data/fbank/cmvn.json  （torch>=2.2.0）"
echo "    ⚠ 元数据 sample_rate 改 16000；use_eos_bos=0、勿加 add_blank；与 V1 元数据 diff 一遍"

echo "[7] 验收：把 model-steps-3.onnx 放回 $REPO/models/zh-en/ 后跑"
echo "    python $REPO/scripts/ruler.py --n 100  对比基线（out/ruler_baseline.log：体温45%/为您自动接听59%/21床3%）"
