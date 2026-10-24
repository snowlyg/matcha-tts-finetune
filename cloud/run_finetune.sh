#!/usr/bin/env bash
# 云端/GPU 机一键微调（骨架）。运行前填好下面的路径，并确保 icefall 已 clone、
# piper_phonemize 已装、自录语料已就位。各 icefall 脚本的精确参数以本机版本 --help 为准。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ICEFALL="${ICEFALL:-$HOME/icefall}"
CORPUS="${CORPUS:-$REPO/data/corpus}"          # 含 wav/ 与 transcripts.tsv
TTS="$ICEFALL/egs/baker_zh/TTS"
PY="${PY:-python3}"

echo "[0] 检查词表特殊 token：作者 tokens.txt 无 _ pad / 无 id0 blank，"
echo "    与 icefall matcha/tokenizer.py 不一致会 KeyError —— 先核对再训。"

echo "[1] 语料 -> 原始 cuts"
"$PY" "$REPO/scripts/prepare_data.py" --corpus "$CORPUS" --out "$REPO/data"

echo "[2] 钉死作者词表（不要跑 generate_tokens.py）"
cp "$REPO/models/zh-en/tokens.txt" "$TTS/data/tokens.txt"

echo "[3] 前端：jieba + lazy_pinyin(tone_sandhi) + piper -> c.tokens"
for split in train valid test; do
  "$PY" "$TTS/local/convert_text_to_tokens.py" \
    --in-file  "$REPO/data/fbank/baker_zh_cuts_${split}.jsonl.gz" \
    --out-file "$REPO/data/fbank/baker_zh_cuts_${split}.tok.jsonl.gz"
done

echo "[4] 特征 + cmvn（参数按本机 icefall 调整；采样率对齐 V1=16k）"
echo "    手动跑: $TTS/local/compute_fbank_baker_zh.py 与 compute_fbank_statistics.py"

echo "[5] 热启动：把 V1 pytorch_model.bin 放成 $TTS/matcha/exp-1/epoch-1.pt，然后"
echo "    python $TTS/matcha/train.py --exp-dir $TTS/matcha/exp-1 --start-epoch 2 \\"
echo "      --num-epochs 30 --max-duration 1200 --bucketing-sampler 1 --num-workers 4 --world-size 1"
echo "    （12GB 显存把 --max-duration 降到 200-400）"

echo "[6] 导出：python $TTS/matcha/export_onnx.py --exp-dir $TTS/matcha/exp-1 --epoch <N> \\"
echo "      --tokens $TTS/data/tokens.txt --cmvn $TTS/data/fbank/cmvn.json  （torch>=2.2.0）"

echo "[7] 验收：把 model-steps-3.onnx 放回 $REPO/models/zh-en/ 后跑"
echo "    python $REPO/scripts/ruler.py --n 100  对比微调前后错率"
