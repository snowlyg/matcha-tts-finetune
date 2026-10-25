#!/usr/bin/env bash
# 云端/GPU 机一键微调（骨架）。运行前填好下面的路径，并确保 icefall 已 clone、
# requirements-tts.txt + piper_phonemize 已装、自录语料已就位。
# 各 icefall 脚本的精确参数以本机版本 --help 为准；四大阻塞已核实，
# 结论与依据见 docs/author-model-facts.md、操作细节见 docs/icefall-finetune-flow.md。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ICEFALL="${ICEFALL:-$HOME/icefall}"
# 配方脚本靠 `import icefall`（clone 不等于可导入）；monotonic_align 需先编译：
#   cd "$ICEFALL/egs/baker_zh/TTS/matcha/monotonic_align" && python setup.py build_ext --inplace
export PYTHONPATH="$ICEFALL:${PYTHONPATH:-}"
CORPUS="${CORPUS:-$REPO/out/corpus1k}"      # 含 wav/ 与 transcripts.tsv（老师合成的语料；自录则指 data/corpus）

# [0] icefall 必需的五处改动：正路是 clone 我们的 fork 分支 matcha-zh-en-16k（见
#     docs/icefall-finetune-flow.md §0），下面各处 sed 都带 grep 守卫、在那份树上全部是 no-op，
#     留着只为"上游裸 clone + 没打补丁"的场合仍能跑通。唯一 sed 补不了的是第 5 处导出守卫
#     （export_onnx.py 用 --cmvn 覆写 model.mel_*）→ 缺了就必须走 fix_mel_buffer 覆写 ckpt。
for probe in "intersperse_blank=False:matcha/train.py" \
             "sampling_rate=16000:local/compute_fbank_baker_zh.py" \
             'normalized_text", None:local/convert_text_to_tokens.py' \
             '"sample_rate": 16000:matcha/export_onnx.py' \
             "overriding with --cmvn:matcha/export_onnx.py"; do
  pat="${probe%%:*}"; f="${probe##*:}"
  grep -qF -- "$pat" "$TTS/$f" || echo "[warn] icefall 缺补丁：$f 里没有 $pat"
done
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
  sed -i 's/intersperse_blank=True/intersperse_blank=False/' "$TTS/matcha/train.py" || true
echo "    已确认: $(grep -n 'intersperse_blank=' "$TTS/matcha/train.py")"

echo "[3] 前端：jieba + lazy_pinyin(tone_sandhi) + piper -> c.tokens"
cd "$TTS"
mkdir -p data/manifests data/fbank
# 3a) 我们的分片 cuts 合并成官方要求的单一 raw cutset（顺序 train→valid→test，
#     便于 [3d] 用 lhotse subset --first/--last 原样复刻切分）
"$PY" - "$REPO" <<'PYEOF'
import sys
from pathlib import Path
from lhotse import CutSet
d = Path(sys.argv[1]) / "data/fbank"
cs = None
for s in ("train", "valid", "test"):
    c = CutSet.from_file(str(d / f"baker_zh_cuts_{s}.jsonl.gz"))
    cs = c if cs is None else cs + c
cs.to_file("data/manifests/baker_zh_cuts_raw.jsonl.gz")
print("merged cuts:", len(cs))
PYEOF
# 3b) 官方脚本读 supervision.normalized_text（由 `lhotse prepare baker-zh` 生成），
#     自录/合成语料只有 .text → 加回退。lhotse 的 custom 字段不随 to_file 序列化，
#     所以注入 normalized_text 这条路已验证走不通，只能改这一行。
grep -q 'getattr(c.supervisions\[0\], "normalized_text"' local/convert_text_to_tokens.py || \
  sed -i 's|text = c.supervisions\[0\].normalized_text|text = getattr(c.supervisions[0], "normalized_text", None) or c.supervisions[0].text|' \
    local/convert_text_to_tokens.py
"$PY" local/convert_text_to_tokens.py \
  --in-file  data/manifests/baker_zh_cuts_raw.jsonl.gz \
  --out-file data/manifests/baker_zh_cuts.jsonl.gz

echo "[4] 特征(16k) + 切分 + cmvn"
# 4a) V1 是 16k（ONNX metadata sample_rate=16000）；baker 配方硬编码 22050，
#     不改会把 16k 语料插值上采样、特征域偏离 V1，且**不报错**（静默训坏）。
grep -q "sampling_rate=22050" local/compute_fbank_baker_zh.py && \
  sed -i 's/22050/16000/g' local/compute_fbank_baker_zh.py || true
"$PY" local/compute_fbank_baker_zh.py --num-jobs 8
"$PY" local/validate_manifest.py data/fbank/baker_zh_cuts.jsonl.gz
# 4b) 切分：三片数量从 [1] 的产物里读，不写死
N_TR=$(gunzip -c "$REPO/data/fbank/baker_zh_cuts_train.jsonl.gz" | wc -l)
N_VA=$(gunzip -c "$REPO/data/fbank/baker_zh_cuts_valid.jsonl.gz" | wc -l)
N_TE=$(gunzip -c "$REPO/data/fbank/baker_zh_cuts_test.jsonl.gz" | wc -l)
N_TAIL=$((N_VA + N_TE))
lhotse subset --last  $N_TAIL data/fbank/baker_zh_cuts.jsonl.gz data/fbank/vt.jsonl.gz
lhotse subset --first $N_VA   data/fbank/vt.jsonl.gz data/fbank/baker_zh_cuts_valid.jsonl.gz
lhotse subset --last  $N_TE   data/fbank/vt.jsonl.gz data/fbank/baker_zh_cuts_test.jsonl.gz
lhotse subset --first $N_TR   data/fbank/baker_zh_cuts.jsonl.gz data/fbank/baker_zh_cuts_train.jsonl.gz
rm data/fbank/vt.jsonl.gz
# 4c) cmvn：mean/std 既要喂训练，也要覆写热启动 ckpt 里的 mel buffer
"$PY" local/compute_fbank_statistics.py \
  data/fbank/baker_zh_cuts_train.jsonl.gz data/fbank/cmvn.json
cat data/fbank/cmvn.json

# 4d) blank 自检：train.py 的 data_args.add_blank 默认 true，但真正插 blank 的是
#     prepare_input 的 intersperse_blank（[2b] 已关）。这里直接看 tokens 落在哪个字段、
#     有没有 0（=`_`）被预插进去——猜 schema 不如看一眼。
gunzip -c data/fbank/baker_zh_cuts_train.jsonl.gz | head -1 | "$PY" -m json.tool | head -60 || true

echo "[5] 热启动：基座必须是 pytorch_model_sherpa.bin（与现网 steps-3 ONNX 权重一致）"
# cmvn.json 是标量对：{"fbank_mean": ..., "fbank_std": ...}（不是 80 维向量）。
# ⚠ 实测（2026-10-25）自算 16k cmvn ≈ (-6.21, 2.90)，而 V1 ckpt 的 mel buffer 是 (-2.95, 5.45)
#   —— 差得很多，说明作者用的 fbank 实现/数据分布与我们不同。此时**默认不覆写 buffer**
#   （保住预训练权重所在的 mel 域），靠 [5b] 的 loss 起点判断；只有 loss 明显异常才试 --mel-mean/--mel-std。
mkdir -p matcha/exp-1
"$PY" "$REPO/scripts/make_warmstart_ckpt.py" \
  --bin "$REPO/models/zh-en/pytorch_model_sherpa.bin" \
  --tokens "$TTS/data/tokens.txt" \
  --out matcha/exp-1/epoch-1.pt
# 备选（确认需要换域时才用）：追加
#   --mel-mean $("$PY" -c "import json;print(json.load(open('data/fbank/cmvn.json'))['fbank_mean'])") \
#   --mel-std  $("$PY" -c "import json;print(json.load(open('data/fbank/cmvn.json'))['fbank_std'])")

echo "[5b] 前置门控：热启动 vs 随机初始化对照，别直接开 --num-epochs 1000（那是从零训练用的默认值）"
# 判据不能用绝对 loss：diff_loss 的量纲 = 1 + var(归一化后 mel)，换 cmvn 就换尺度。
# 实测基线（2026-10-25，0.92h 语料、md 60）：随机 batch0 = dur 3.28/prior 1.49/diff 2.95/tot 7.72；
# 热启动 0 step = 1.79/1.18/0.87/3.84、batch10 diff 已 0.40。三个头同时低于随机才算权重真吃进去
# （icefall load_checkpoint 默认 strict=False，键名没对上不报错、静默留随机权重）。
mkdir -p matcha/exp-scratch
"$PY" matcha/train.py --exp-dir matcha/exp-scratch --start-epoch 1 --num-epochs 2 \
  --max-duration 60 --bucketing-sampler 1 --num-workers 4 --num-buckets 10 \
  --save-every-n 100 2>&1 | grep "batch 0," || true
rm -rf matcha/exp-scratch
# 三个坑（实测 2026-10-25）：
#  1) --num-buckets 默认 30，必须 ≤ valid cuts 数，否则 lhotse 直接 assert 失败（小语料必踩）。
#  2) --num-epochs 是"最后一个 epoch 的索引（含）"，且 --start-epoch N 加载 epoch-(N-1).pt。
#     实测：--start-epoch 2 --num-epochs 2 真跑了一轮 epoch 2；--num-epochs 8 跑到 Epoch 8。
#     跑 N 轮 → --num-epochs $((start+N-1))；断点续训 → --start-epoch <已存epoch+1>。
#  3) 没有 --log-interval/--valid-interval 参数（train.py:246 写死 10/1500）。语料 <1k 条时
#     --max-duration 600 ≈ 每 epoch 6 个 batch → 一行 step loss 都打不出来；
#     想读 loss 就把 --max-duration 压到 60（~55 batch/epoch、~0.1 秒/batch ≈ 5 秒/epoch）。
#     另外 `| tee x.log | grep | head -N` 会让 head 关管道 → python 收 SIGPIPE 当场死。
"$PY" matcha/train.py --exp-dir matcha/exp-1 --start-epoch 2 --num-epochs 8 \
  --max-duration 60 --bucketing-sampler 1 --num-workers 4 --num-buckets 10 --save-every-n 100
# 门控过了再投正式微调。**轮数交给 ruler 判，不要拿 valid 曲线当早停依据**：
# md 60 ≈ 5–8 秒/epoch、ckpt 每个 ~214MB；save-every-n 20 → 每 20 轮一份候选。
# ⚠ 实测（2026-10-25，0.92h 语料、lr 1e-4）四个验证点 1.791/1.701/1.801/1.749 来回跳，
#   diff_loss 非单调（0.267→0.294→0.350→0.280）—— 19 条 valid 的抖动量级就等于曲线变化量级，
#   **判不出过拟合拐点**（我据 3 个点判过"57 见底"，第 4 点就推翻了）。所以跑满、多留 ckpt，
#   到 [6] 各导一份 ONNX 用错率选。valid_interval 写死 1500 batch，md 60 下 ≈ 27 epoch 才一点。
# nohup 是为了 ssh 掉线不杀训练；日志单独 grep，别接 head：
# 断点续训：--start-epoch <最新已存epoch+1>（它加载 epoch-(start-1).pt，优化器态一起恢复）
nohup "$PY" matcha/train.py --exp-dir matcha/exp-1 --start-epoch 2 --num-epochs 302 \
  --max-duration 60 --bucketing-sampler 1 --num-workers 4 --num-buckets 10 \
  --save-every-n 20 > /tmp/ft.log 2>&1 &
sleep 5
# 判重不要用 pgrep -c：DataLoader 的 num_workers 个子进程继承同一条 cmdline，会被一起数进去
# （实测 1 个训练 + 4 workers 就报 7）。真正只可能有一个的是**占卡进程**：
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 计算进程必须只有 1 行
tail -3 /tmp/ft.log
#   （12GB 显存把 --max-duration 降到 200-400；24G 上 md 1200 也放得下）

echo "[6] 导出 ONNX（接口已实测：只有 --epoch/--exp-dir/--tokens/--cmvn 四个参数）"
#   ⚠ 默认值很危险：--exp-dir 默认 matcha/exp-new-3、--epoch 默认 2000 → 必须显式覆盖，
#     否则会去加载不存在的 ckpt。--cmvn 的 help 文案写错成 "Path to vocabulary"，按名字给即可。
"$PY" matcha/export_onnx.py --help | tail -8
# 元数据里的 sample_rate 写死 22050 必须改 16000（现网 V1 即 16000，端侧按它定时长/采样）。
# 实测行号：:190 sample_rate=22050（要改）、:196 use_eos_bos=0（源码不用动；最终由覆盖步骤定值）。
grep -q '"sample_rate": 22050' matcha/export_onnx.py && \
  sed -i 's/"sample_rate": 22050/"sample_rate": 16000/' matcha/export_onnx.py || true
echo "    已确认: $(grep -n 'sample_rate\|use_eos_bos' matcha/export_onnx.py)"
#   导出（需 torch>=2.2.0）：验证曲线判不出拐点，所以每个候选 epoch 都导一份，用错率选。
#   ⚠⚠ 导出物**写在当前工作目录、文件名写死 model-steps-2..6.onnx**，不在 --exp-dir 里，
#      而且换 epoch 再导会**原地静默覆盖**（实测连导 40/60/100 三轮后只剩 100 那批）。
#      → 每导一个 epoch 必须立刻改名存档。
mkdir -p cand
for e in 40 60 100 140 180; do
  "$PY" matcha/export_onnx.py --exp-dir matcha/exp-1 --epoch "$e" \
    --tokens data/tokens.txt --cmvn data/fbank/cmvn.json > "/tmp/export-$e.log" 2>&1 && \
    cp model-steps-3.onnx "cand/steps-3-ep$e.onnx" || echo "[warn] epoch $e 导出失败，看 /tmp/export-$e.log"
done
ls -l cand/
#   键集合与现网对不上（实测导出 15 键 vs V1 12 键：language 变 Chinese、voice 整个缺失，
#   多 jieba/has_espeak/maintainer…）。language/voice 是端侧前端会读的键，逐键判断会漏
#   → 整份覆盖 V1 的 metadata_props（fix_onnx_meta.py 随本脚本一起 scp 上云）。
#     --src 用上面 find 的实际路径填，别用通配（steps-2..6 五份都在）：
#     python "$REPO/scripts/fix_onnx_meta.py" --ref "$REPO/models/zh-en/model-steps-3.onnx" \
#       --src <find 到的 model-steps-3.onnx> --out model-steps-3.fixed.onnx
#   默认 --keep num_ode_steps（该键描述这张图跑了几步 ODE，跟着参照走会让 steps-5 说谎）。

echo "[6b] 出声门控（**必须在投长训之前跑**，不是等 300 轮跑完）"
# 首轮的血：loss 从 3.84 一路降到 1.46 很"健康"，ruler 却是 92.2% vs 基线 25.8%——
# CFM/dur loss 对 mel 归一化域错误**完全不敏感**（它就是在歪域里拟合）。所以门控只能看声音。
# 在实例上跑 --n 10 做"有没有救"的二分判定（SenseVoice 已在 venv、CPU 推理，几分钟）；
# 最终对错率必须在 Mac 上与基线同机比（[7]），跨机器的 ASR 不同实现不可比。
mkdir -p gate
ln -sf "$REPO/models/zh-en/vocos-16khz-univ.onnx" gate/
# 取手边最早的一份候选（两种命名都认：[6] 的 steps-3-ep*.onnx、session_fix_mel 的 steps-3-melfix-ep*.onnx）
GATE_SRC="${GATE_SRC:-$(ls cand/steps-3-*.onnx 2>/dev/null | head -1)}"
[ -n "$GATE_SRC" ] || { echo "!! gate 没有候选 onnx，先跑 [6] 导出"; exit 1; }
echo "   门控用：$GATE_SRC"
cp "$GATE_SRC" gate/model-steps-3.onnx
TTS_MODEL_DIR=gate "$PY" "$REPO/scripts/ruler.py" --n 10 --threads 4 | tail -8
# 建议的节奏：先 --num-epochs 42 短训一轮（约 5 分钟）→ 跑这一步 → 错了就停下改域，
# 对了再 --start-epoch 43 续到目标轮数。别等 300 轮跑完才发现整条 run 是废的。

echo "[7] 终验：把 model-steps-3.fixed.onnx 改名放回 $REPO/models/zh-en/model-steps-3.onnx 后跑"
echo "    python $REPO/scripts/ruler.py --n 100  对比基线（out/ruler_baseline.log：体温45%/为您自动接听59%/21床3%）"
echo "    必须同一台机器、同一套 SenseVoice 运行时才可比；RTF 也在这一步一起看。"
