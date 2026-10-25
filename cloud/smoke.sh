#!/usr/bin/env bash
# 全链冒烟：换云厂商/新实例后**第一条要跑的命令**。它不等真语料、不等长训练，
# 用 56 句合成波形把「数据→特征→切分→热启动→训练→导出→出声」整条管线走一遍，
# 每步只验证"这条链在这台机器上能不能产出正确东西"。
#
# 为什么值得单独一个脚本（2026-10-25 的教训）：首轮跑完 300 epoch、loss 一路降到 1.46
# 看起来完全健康，导出后 ruler 92.2%——真因是导出用的反归一化 buffer 与训练归一化不是同一对。
# **loss 对这类域错误完全不敏感，只有出声门控能拦住**。门控必须放在投长训之前，
# 所以把它固化成这一步；另外本脚本**只测管线**，不代表模型质量（音频门控用 V1 本身，
# 验的是"这台机器能合成+能 ASR 回读"，不是"训练有效果"）。
#
# 用法（实例上）：
#   . /root/env_gpu.sh                      # 重连后第一件事
#   bash $REPO/cloud/smoke.sh               # 约 3~6 分钟（CPU 导出慢些）
#   KEEP_SMOKE=1 bash $REPO/cloud/smoke.sh  # 保留冒烟产物（默认跑完清掉 matcha/exp-smoke）
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
ICEFALL="${ICEFALL:-$HOME/icefall}"
TTS="${TTS:-$ICEFALL/egs/baker_zh/TTS}"
PY="${PY:-python3}"
SMOKE_CORPUS="$REPO/out/smoke_corpus"
FAIL=0
ok()   { echo "  [ok]   $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
info() { echo "  [info] $*"; }

echo "=============================================================="
echo "[1] 会话环境（ssh 重连后最容易踩：venv 没激活就退回 conda 的 python3）"
echo "=============================================================="
"$PY" - <<'PYEOF' || bad "python/venv/torch 不可用"
import sys
print("  python   :", sys.executable)
import torch
print("  torch    :", torch.__version__, " cuda:", torch.cuda.is_available(),
      " device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
import k2, lhotse, onnx, soundfile, sherpa_onnx
print("  k2/lhotse/onnx/sherpa_onnx: ok  (lhotse", lhotse.__version__ + ")")
PYEOF
[ -f "$REPO/.venv/bin/activate" ] || bad "没有 $REPO/.venv（先跑 cloud/bootstrap_gpu.sh）"
"$PY" -c "import icefall" 2>/dev/null \
  && ok "PYTHONPATH 里有 icefall（clone 不等于可导入）" \
  || bad "import icefall 失败：先 . \$REPO/cloud/env_gpu.sh（该文件已把 PYTHONPATH 设好）"
if [ "${CUDA_VISIBLE_DEVICES+x}" = "x" ]; then
  info "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES（已显式设置）"
else
  info "CUDA_VISIBLE_DEVICES 未设置（用全部 GPU）"
fi
"$PY" -c "import piper_phonemize" 2>/dev/null && ok "piper_phonemize（前端 G2P 依赖）" \
  || bad "piper_phonemize 缺失 → bootstrap_gpu.sh 的 pip 段没跑完"
if [ -d "$TTS/matcha/monotonic_align" ] && \
   ls "$TTS"/matcha/monotonic_align/*.so >/dev/null 2>&1; then
  ok "monotonic_align 已编译（setup.py build_ext --inplace）"
else
  bad "monotonic_align 未编译：cd $TTS/matcha/monotonic_align && $PY setup.py build_ext --inplace"
fi

echo "=============================================================="
echo "[2] icefall 六处必需改动（正路是 fork 分支 matcha-zh-en-16k；缺任一都会静默训坏）"
echo "=============================================================="
cd "$TTS" || { echo "!! 没有 $TTS，先跑 bootstrap"; exit 1; }
for probe in "intersperse_blank=False:matcha/train.py:不插 blank，否则与现网推理语义不符" \
             "sampling_rate=16000:local/compute_fbank_baker_zh.py:22050 会把 16k 语料静默上采样" \
             'normalized_text", None:local/convert_text_to_tokens.py:合成/自录语料没有 normalized_text' \
             '"sample_rate": 16000:matcha/export_onnx.py:端侧按这个键定时长/采样' \
             "overriding with --cmvn:matcha/export_onnx.py:用训练那对覆写反归一化 buffer（首轮 92.2% 的真因）" \
             "if p.requires_grad]:matcha/train.py:没有它就调不了 lr、冻结不了参数"; do
  pat="${probe%%:*}"; rest="${probe#*:}"
  f="${rest%%:*}"; why="${rest#*:}"
  if grep -qF -- "$pat" "$f" 2>/dev/null; then
    ok "$f ← $pat"
  else
    bad "$f 缺改动「$pat」（$why）→ 用 $REPO/patches/icefall-baker-zh-16k.patch：
           (cd $ICEFALL && patch -p1 -N --forward < $REPO/patches/icefall-baker-zh-16k.patch)"
  fi
done

echo "=============================================================="
echo "[3] 资产：V1 基座 + 词表 + SenseVoice 尺子"
echo "=============================================================="
for f in "$REPO/models/zh-en/model-steps-3.onnx" \
         "$REPO/models/zh-en/vocos-16khz-univ.onnx" \
         "$REPO/models/zh-en/pytorch_model_sherpa.bin" \
         "$REPO/models/vocab_tts.txt" \
         "$REPO/models/sense-voice/model.int8.onnx" \
         "$REPO/models/sense-voice/tokens.txt"; do
  [ -s "$f" ] && ok "$(basename "$f") ($(du -h "$f" | cut -f1))" || bad "缺 $f"
done

echo "=============================================================="
echo "[4] 数据链：合成 56 句占位波形 -> cuts -> tokens -> fbank -> cmvn"
echo "=============================================================="
if [ ! -f "$SMOKE_CORPUS/transcripts.tsv" ]; then
  mkdir -p "$SMOKE_CORPUS/wav"
  "$PY" - "$SMOKE_CORPUS" <<'PYEOF' || { bad "生成冒语音料失败"; exit 1; }
import itertools
import os, sys
import numpy as np
import soundfile as sf

d = sys.argv[1]
os.makedirs(os.path.join(d, "wav"), exist_ok=True)
# 短语里没有一条来自训练语料；内容不重要（这一步验管线），但别和验收尺子撞句
PHRASES = [
    "请把袖子往上挽一挽", "血压一百二十六十八", "零点儿五毫克一次",
    "下午四点四十分换药", "三十七度八需要复诊", "里屋那位病人家属来了",
    "九床的液体输完了", "帮他量个体温吧", "这个药要慢慢儿喝",
    "咱们不着急慢慢说", "七十八床呼叫过三次", "大夫一会儿就过来",
    "体重五十二点三千克", "先坐这儿歇一会儿", "药盒上有用法用量",
    "护士站电话零一二三",
]
# 冒烟只需要"够切出 valid/test"的量，不需要语义：16 条真人话术 + 40 条模板×数字变体 = 56 条
NUM = ["三点二十分", "七点四十五分", "十一点零五分", "十四点三十分",
       "十九点十五分", "二十一点四十分", "二点五十分", "四点十分"]
UNIT = ["复查", "换药", "给药", "抽血", "输氧"]
WINDOW = ["三号窗口", "五号窗口", "七号窗口"]
for k, (a, b) in enumerate(itertools.product(NUM, UNIT)):
    PHRASES.append(f"{WINDOW[k % 3]}安排在{a}{b}")
rng = np.random.default_rng(7)
rows = []
for i, t in enumerate(PHRASES):
    dur = 0.9 + (i % 5) * 0.25
    n = int(16000 * dur)
    tt = np.arange(n) / 16000.0
    f0 = 130.0 + 40.0 * (i % 4)
    x = np.zeros(n)
    for h in range(1, 8):                       # 谐波叠加＝类浊音，别用纯静音（cmvn 会退化）
        x += np.sin(2 * np.pi * f0 * h * tt) / h
    x += 0.25 * rng.standard_normal(n)          # 加噪＝类擦音，让 fbank 有内容
    env = np.hanning(n) ** 0.35
    x = (x * env / (np.abs(x * env).max() + 1e-9) * 0.28).astype(np.float32)
    uid = f"s{i + 1:04d}"
    sf.write(os.path.join(d, "wav", uid + ".wav"), x, 16000, "PCM_16")
    rows.append(f"{uid}\t{t}")
open(os.path.join(d, "transcripts.tsv"), "w").write("\n".join(rows) + "\n")
print("wrote", len(rows), "wav")
PYEOF
fi
# ⚠ 必须显式给 --split：默认 "0.96,0.02,0.02" 在小语料上会把 valid 切成 **0 条**
#   （56 条 + 0.7/0.15/0.15 → 39/8/9；默认比例下 16 条会切成 15/0/1，实测踩过），而 valid 空会让训练与导出连环崩，
#   看起来像"这台机器不行"其实是冒烟自己的样本量问题（2026-10-25 在这台 4090 上踩过）。
"$PY" "$REPO/scripts/prepare_data.py" --corpus "$SMOKE_CORPUS" --out "$REPO/data" \
  --split 0.7,0.15,0.15 >/tmp/smoke-prep.log 2>&1 \
  && ok "prepare_data（g2p 前端可用、cuts 可序列化）" || { bad "prepare_data 失败，看 /tmp/smoke-prep.log"; tail -15 /tmp/smoke-prep.log; }
mkdir -p data/manifests
"$PY" - "$REPO" <<'PYEOF' || bad "合并 raw cuts 失败"
import sys
from pathlib import Path
from lhotse import CutSet
d = Path(sys.argv[1]) / "data" / "fbank"
cs = None
for s in ("train", "valid", "test"):
    c = CutSet.from_file(str(d / f"baker_zh_cuts_{s}.jsonl.gz"))
    cs = c if cs is None else cs + c
cs.to_file("data/manifests/baker_zh_cuts_raw.jsonl.gz")
print("  merged cuts:", len(cs))
PYEOF
"$PY" local/convert_text_to_tokens.py \
  --in-file data/manifests/baker_zh_cuts_raw.jsonl.gz \
  --out-file data/manifests/baker_zh_cuts.jsonl.gz >/tmp/smoke-t2t.log 2>&1 \
  && ok "convert_text_to_tokens（jieba+lazy_pinyin+piper 同源前端）" \
  || { bad "convert_text_to_tokens 失败，看 /tmp/smoke-t2t.log"; tail -15 /tmp/smoke-t2t.log; }
"$PY" local/compute_fbank_baker_zh.py --num-jobs 4 >/tmp/smoke-fbank.log 2>&1 \
  && ok "compute_fbank（16k）" || { bad "compute_fbank 失败，看 /tmp/smoke-fbank.log"; tail -15 /tmp/smoke-fbank.log; }
N_TR=$(gunzip -c "$REPO/data/fbank/baker_zh_cuts_train.jsonl.gz" | wc -l)
N_VA=$(gunzip -c "$REPO/data/fbank/baker_zh_cuts_valid.jsonl.gz" | wc -l)
N_TE=$(gunzip -c "$REPO/data/fbank/baker_zh_cuts_test.jsonl.gz" | wc -l)
N_TAIL=$((N_VA + N_TE))
lhotse subset --last $N_TAIL data/fbank/baker_zh_cuts.jsonl.gz data/fbank/vt.jsonl.gz >/dev/null 2>&1
lhotse subset --first $N_VA data/fbank/vt.jsonl.gz data/fbank/baker_zh_cuts_valid.jsonl.gz >/dev/null 2>&1
lhotse subset --last $N_TE data/fbank/vt.jsonl.gz data/fbank/baker_zh_cuts_test.jsonl.gz >/dev/null 2>&1
lhotse subset --first $N_TR data/fbank/baker_zh_cuts.jsonl.gz data/fbank/baker_zh_cuts_train.jsonl.gz >/dev/null 2>&1
rm -f data/fbank/vt.jsonl.gz
ok "lhotse 切分 train/valid/test = $N_TR/$N_VA/$N_TE"
"$PY" local/compute_fbank_statistics.py data/fbank/baker_zh_cuts_train.jsonl.gz data/fbank/cmvn.json >/dev/null 2>&1 \
  && ok "cmvn: $(cat data/fbank/cmvn.json)" || bad "compute_fbank_statistics 失败"
NB=$("$PY" -c "print(max(1, $N_VA - 1))")
info "--num-buckets 用 $NB（必须 ≤ valid cuts=$N_VA，否则 lhotse 直接 assert）"

echo "=============================================================="
echo "[5] 词表 + 热启动 ckpt（键名/行数不对会 strict=False 静默留随机权重）"
echo "=============================================================="
mkdir -p data
"$PY" "$REPO/scripts/make_train_tokens.py" --vocab "$REPO/models/vocab_tts.txt" \
  --out data/tokens.txt >/tmp/smoke-tokens.log 2>&1 \
  && ok "tokens.txt: $(wc -l < data/tokens.txt) 行" || { bad "make_train_tokens 失败"; tail -10 /tmp/smoke-tokens.log; }
mkdir -p matcha/exp-smoke
"$PY" "$REPO/scripts/make_warmstart_ckpt.py" \
  --bin "$REPO/models/zh-en/pytorch_model_sherpa.bin" \
  --tokens data/tokens.txt \
  --out matcha/exp-smoke/epoch-1.pt >/tmp/smoke-warm.log 2>&1 \
  && ok "热启动 epoch-1.pt" || { bad "make_warmstart_ckpt 失败，看 /tmp/smoke-warm.log"; tail -15 /tmp/smoke-warm.log; }

echo "=============================================================="
echo "[6] A/B 前置门控：热启动 vs 随机初始化（三头必须同时低于随机）"
echo "=============================================================="
"$PY" - <<'PYEOF' || true
import json
m = json.load(open("data/fbank/cmvn.json"))
print(f"  训练归一化用 ({m['fbank_mean']:.4f}, {m['fbank_std']:.4f}) —— 导出反归一化必须是这同一对")
PYEOF
SCRATCH="$(mktemp -d)"; mkdir -p "$SCRATCH/exp-scratch"
"$PY" matcha/train.py --exp-dir "$SCRATCH/exp-scratch" --start-epoch 1 --num-epochs 1 \
  --max-duration 30 --bucketing-sampler 1 --num-workers 2 --num-buckets "$NB" \
  --save-every-n 100 > "$SCRATCH/random.log" 2>&1
grep -m1 "batch 0," "$SCRATCH/random.log" | sed 's/^/  随机 /' || info "随机基线没打出 loss 行（batch 太少，看 $SCRATCH/random.log）"
"$PY" matcha/train.py --exp-dir matcha/exp-smoke --start-epoch 2 --num-epochs 4 \
  --max-duration 30 --bucketing-sampler 1 --num-workers 2 --num-buckets "$NB" \
  --save-every-n 1 > /tmp/smoke-train.log 2>&1
grep -m1 "batch 0," /tmp/smoke-train.log | sed 's/^/  热启动 /' || { bad "热启动训练没有 loss 行，看 /tmp/smoke-train.log"; tail -20 /tmp/smoke-train.log; }
if [ -f matcha/exp-smoke/epoch-4.pt ]; then ok "epoch-4.pt 已保存"; else bad "没产出 matcha/exp-smoke/epoch-4.pt"; fi
rm -rf "$SCRATCH"

echo "=============================================================="
echo "[7] 导出 + 结构校验（三处必须对：sample_rate、mel 那对、元数据键集合）"
echo "=============================================================="
# 导出物**写在 CWD、文件名写死 model-steps-2..6.onnx**（不在 --exp-dir 里），
# 所以这里也必须在 $TTS 下导，导完立刻改名挪走 —— 与 run_finetune.sh [6] 同一套动作。
mkdir -p /tmp/smoke-export
EXPORTED=/tmp/smoke-export/model-steps-3.onnx
[ -f matcha/exp-smoke/epoch-4.pt ] \
  && "$PY" matcha/export_onnx.py --exp-dir matcha/exp-smoke --epoch 4 \
       --tokens data/tokens.txt --cmvn data/fbank/cmvn.json > /tmp/smoke-export.log 2>&1 \
  && cp model-steps-3.onnx "$EXPORTED" && rm -f model-steps-{2,4,5,6}.onnx \
  && [ -s "$EXPORTED" ] \
  && ok "导出 model-steps-3.onnx ($(du -h "$EXPORTED" | cut -f1))" \
  || { bad "导出失败，看 /tmp/smoke-export.log"; tail -20 /tmp/smoke-export.log; }
# 导出守卫有没有真的生效（首轮 92.2% 的病灶就在这里）
if grep -q "overriding with --cmvn" /tmp/smoke-export.log; then
  ok "导出守卫生效：日志里有 overriding with --cmvn"
else
  info "日志里没有覆写告警 → ckpt 的 buffer 与 --cmvn 一致，或用的是没打第 5 处补丁的 icefall"
fi
"$PY" - "$REPO/models/zh-en/model-steps-3.onnx" "$EXPORTED" "$TTS/data/fbank/cmvn.json" <<'PYEOF' || bad "导出物结构校验未通过"
import onnx, numpy as np, sys, json
ref, src, cmvn = sys.argv[1], sys.argv[2], sys.argv[3]
cm = json.load(open(cmvn))
mean, std = cm["fbank_mean"], cm["fbank_std"]
g = onnx.load(src)
md = {k.key: k.value for k in g.metadata_props}
assert md.get("sample_rate") == "16000", md.get("sample_rate")
print(f"  [ok]   metadata sample_rate=16000, 共 {len(md)} 键")
assert "add_blank" not in md, "元数据里出现 add_blank，端侧会插 blank"
print("  [ok]   无 add_blank 键")
vals = np.concatenate([onnx.numpy_helper.to_array(i).ravel().astype(float)
                       for i in g.graph.initializer
                       if onnx.numpy_helper.to_array(i).dtype.kind == "f"])
hit_m = int(np.isclose(vals, mean, atol=1e-4).sum()); hit_s = int(np.isclose(vals, std, atol=1e-4).sum())
v1_m = int(np.isclose(vals, -2.9522, atol=1e-4).sum()); v1_s = int(np.isclose(vals, 5.4468, atol=1e-4).sum())
print(f"  mel 常数：训练那对({mean:.4f},{std:.4f})命中 {hit_m}/{hit_s}，V1 那对命中 {v1_m}/{v1_s}")
assert hit_m and hit_s, "导出物里没有训练用的 mel 常数 → 归一化/反归一化不同对，出声必坏"
if v1_m or v1_s:
    print("  [warn] V1 那对也还在（多半是残留 buffer，不影响；以训练那对为准）")
refmd = {k.key: k.value for k in onnx.load(ref).metadata_props}
print(f"  [info] 与 V1 键差：多 {sorted(set(md) - set(refmd))} 少 {sorted(set(refmd) - set(md))}"
      f"（发端侧前用 scripts/fix_onnx_meta.py 整份覆盖）")
print("  [ok]   导出物结构校验通过")
PYEOF

echo "=============================================================="
echo "[8] 出声门控：这台机器能不能合成 + ASR 能不能回读（用 V1 本身）"
echo "=============================================================="
# 注意这一步验的是**工具链**，不是训练效果：V1 的错率就是这台机器的基线，
# 正式验收必须与它同机可比（跨机器的 ASR/线程数不同不可比）。
# 两套尺子都跑：DEFAULT 5 句是"病灶短语"（已进训练语料，只证明没训坏）；
# scripts/ruler_heldout.tsv 才是验收用的 held-out 集（与语料零交集，程序校验过）。
[ -s "$REPO/scripts/ruler_heldout.tsv" ] && ok "held-out 尺子在仓（$(grep -vc '^#' "$REPO/scripts/ruler_heldout.tsv") 条）" \
  || bad "缺 $REPO/scripts/ruler_heldout.tsv"
for ph in "" "--phrases-file $REPO/scripts/ruler_heldout.tsv"; do
  "$PY" "$REPO/scripts/ruler.py" --n 4 --threads 4 $ph > /tmp/smoke-ruler-v1.log 2>&1 \
    && { echo "  --- ${ph:-DEFAULT 5 句} ---"; tail -1 /tmp/smoke-ruler-v1.log;
         ok "合成+回读链路可用（上面是 V1 在本机的基线）"; } \
    || { bad "ruler 跑不通，看 /tmp/smoke-ruler-v1.log"; tail -20 /tmp/smoke-ruler-v1.log; }
done

echo "=============================================================="
echo "[9] 元数据整份覆盖（端侧替换前的一步，只做工具自检）"
echo "=============================================================="
"$PY" "$REPO/scripts/fix_onnx_meta.py" \
  --ref "$REPO/models/zh-en/model-steps-3.onnx" \
  --src /tmp/smoke-export/model-steps-3.onnx \
  --out /tmp/smoke-export/model-steps-3.fixed.onnx >/tmp/smoke-meta.log 2>&1 \
  && [ -s /tmp/smoke-export/model-steps-3.fixed.onnx ] \
  && ok "fix_onnx_meta 可用：$(tail -1 /tmp/smoke-meta.log)" || { bad "fix_onnx_meta 失败"; tail -15 /tmp/smoke-meta.log; }

echo "=============================================================="
if [ "${KEEP_SMOKE:-}" = "1" ]; then
  info "KEEP_SMOKE=1 → 保留 matcha/exp-smoke 与 $REPO/data（冒烟语料的产物）"
else
  rm -rf matcha/exp-smoke "$REPO/data" /tmp/smoke-export
  info "已清掉 matcha/exp-smoke、$REPO/data、/tmp/smoke-export（正式跑 [1] 会重新生成）"
fi
echo "[10] 结论"
echo "=============================================================="
if [ "$FAIL" = 0 ]; then
  echo "  冒烟全过：这台机器可以投正式微调。"
  echo "  下一步：把真语料放到 ${CORPUS:-$REPO/out/corpus1k}（wav/ + transcripts.tsv），"
  echo "          编辑并跑 cloud/run_finetune.sh（短训 42 轮 → [6b] 出声门控 → 再续长训）。"
else
  echo "  有 $FAIL 项未过 —— 按上面的 [FAIL] 逐条修，全部修完再投训练。"
  echo "  别跳过冒烟直接长训：首轮就是 loss 一路降、300 轮后才发现导出域错，白烧一天。"
  exit 1
fi
