#!/usr/bin/env bash
# 多臂对照跑训练：**一次开机里只变一个东西**，其余全锁死（同 seed/轮数/warmstart/导出流程）。
# 第一组 A/B 变的是语料语速（已跑完，结论见 docs/icefall-finetune-flow.md §7.3）；
# 默认的第二组 c/d/e 变的是配方（lr、replay 混合、冻结 CFM estimator）。
# 臂表在 `ARMS_DEFAULT`，也可以外部传：`AB_ARMS='f|/root/out/x|--lr 2e-5;g|...'`。
#
# 为什么是这个实验（2026-10-25）：held-out 上三份候选全在 83~86%（V1 基线 24.3% 音口径），
# 且损伤形态是吞字不是读错。用 scripts/duration_gate.py 量出来：ep300 输出时长是 V1 的 0.87、
# ep60 就已经不通过门控。而语料侧实测的"秒/音节随句长单调下降"（7~9 音节 0.285 ≈ V1 的 0.284，
# 16+ 音节 0.240）正好给出同样的 13% 缺口。`flatten_rate.py` 把语料抹平到长句档偏差 0.8%。
# **假设能不能成立，只有对着训一轮才知道**——lr、轮数、种子、结构全锁死，只动语料。
#
# 判读（跑之前就登记，别事后编）：
#   ① duration_gate：B 相对 V1 的中位时长比 ≥0.95 而 A 仍 <0.95 → 假设成立，语料是主因；
#   ② held-out 音口径：B ≤ A 才有意义；若 B 仍远高于 V1 的 24.3%，说明语料语速只是共犯之一，
#      下一步该动的是配方（lr/冻结）而不是继续修语料。
#   两个都过才允许投长训。
#
# 用法（实例上，一次开机跑完；每臂约 8 分钟，两臂 + 导出 ≈ 20 分钟）：
#   . /root/env_gpu.sh
#   bash $REPO/cloud/ab_rate.sh                  # 默认 40 轮
#   AB_EPOCHS=60 bash $REPO/cloud/ab_rate.sh     # 想多留一个点就改这里
# 语料两份都要先在实例上：out/corpus1k（原始）与 out/corpus1k_flat（抹平+复验剔除后）。
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
ICEFALL="${ICEFALL:-$HOME/icefall}"
TTS="${TTS:-$ICEFALL/egs/baker_zh/TTS}"
PY="${PY:-python3}"
EP="${AB_EPOCHS:-40}"
MD="${AB_MAX_DURATION:-60}"
NB="${AB_NUM_BUCKETS:-10}"

# 臂定义：`名字|语料目录|额外 train.py 参数`，一行一臂。
# 第一组 A/B（只换语料语速）已跑完：**假设被证伪**——语音时长比 A 0.992 / B 1.106，
# 整段时长的差别全在固定首尾静音上，内容错率两臂一样烂（详见 flow.md §7.2/§7.3）。
# 所以 c/d/e 三臂**回到原始教师语料**（抹平那步已降级为可选，不再作实验基线），
# 各测一个配方杠杆，除 extra 外完全同构（同 seed、同轮数、同 warmstart）：
#   c 仅降 lr                —— lr 能不能单独解释这次遗忘
#   d 降 lr + V1 自产 replay —— 数据侧回放能不能挡住遗忘
#   e 降 lr + 冻结 CFM estimator —— 结构侧锁住生成主干，只让 encoder/dur 学
ARMS_DEFAULT=(
  "c|$REPO/out/corpus1k|--lr 1e-5"
  "d|$REPO/out/corpus_mix|--lr 1e-5"
  "e|$REPO/out/corpus1k|--lr 1e-5 --freeze decoder.estimator"
)
if [ -n "${AB_ARMS:-}" ]; then
  IFS=';' read -r -a ARMS <<< "$AB_ARMS"        # 覆盖时用 ';' 分隔多臂，便于 shell 里传
else
  ARMS=("${ARMS_DEFAULT[@]}")
fi

cd "$TTS" || { echo "!! 没有 $TTS"; exit 1; }
for spec in "${ARMS[@]}"; do
  d="$(echo "$spec" | cut -d'|' -f2)"
  [ -d "$d/wav" ] || { echo "!! 缺语料 $d（先从 Mac 传上来）"; exit 1; }
done
echo "[ab] 轮数=$EP max-duration=$MD num-buckets=$NB  臂数=${#ARMS[@]}"
for spec in "${ARMS[@]}"; do echo "     $spec"; done
# 训练词表由 run_finetune.sh [2] 生成；新实例上没跑过就得补，否则热启动/训练一起失败
[ -s data/tokens.txt ] || {
  echo "[ab] 缺 data/tokens.txt，用 make_train_tokens.py 生成"
  "$PY" "$REPO/scripts/make_train_tokens.py" --vocab "$REPO/models/vocab_tts.txt" \
    --out data/tokens.txt || { echo "!! 词表生成失败"; exit 1; }
}

# 数据链的产物（$TTS/data/ 下的 cuts/特征/cmvn/tokens）是**双臂共用**的位置，
# 换臂必须重建。先备份现状：这份是首轮 300 轮的域参数，cmvn.json 只此一份、丢了就没法复现旧导出。
if [ ! -e /root/data_pre_ab.tgz ]; then
  tar czf /root/data_pre_ab.tgz data/fbank/cmvn.json data/tokens.txt 2>/dev/null \
    && echo "[ab] 已备份首轮域参数 → /root/data_pre_ab.tgz"
fi

build_data() {          # $1=语料目录  $2=prepare_data 的输出目录
  local corpus="$1" datadir="$2"
  rm -rf "$datadir" data/manifests data/fbank
  mkdir -p "$datadir" data/manifests data/fbank
  "$PY" "$REPO/scripts/prepare_data.py" --corpus "$corpus" --out "$datadir" >/tmp/ab-prep.log 2>&1 \
    || { echo "!! prepare_data 失败 /tmp/ab-prep.log"; return 1; }
  "$PY" - "$datadir" <<'PYEOF' || return 1
import sys
from pathlib import Path
from lhotse import CutSet
d = Path(sys.argv[1]) / "fbank"
cs = None
for s in ("train", "valid", "test"):
    c = CutSet.from_file(str(d / f"baker_zh_cuts_{s}.jsonl.gz"))
    cs = c if cs is None else cs + c
cs.to_file("data/manifests/baker_zh_cuts_raw.jsonl.gz")
print("  merged cuts:", len(cs))
PYEOF
  "$PY" local/convert_text_to_tokens.py \
    --in-file data/manifests/baker_zh_cuts_raw.jsonl.gz \
    --out-file data/manifests/baker_zh_cuts.jsonl.gz >/tmp/ab-t2t.log 2>&1 || {
      echo "!! convert_text_to_tokens 失败"; tail -8 /tmp/ab-t2t.log; return 1; }
  "$PY" local/compute_fbank_baker_zh.py --num-jobs 4 >/tmp/ab-fbank.log 2>&1 || {
      echo "!! compute_fbank 失败"; tail -8 /tmp/ab-fbank.log; return 1; }
  local N_TR N_VA N_TE N_TAIL
  N_TR=$(gunzip -c "$datadir/fbank/baker_zh_cuts_train.jsonl.gz" | wc -l)
  N_VA=$(gunzip -c "$datadir/fbank/baker_zh_cuts_valid.jsonl.gz" | wc -l)
  N_TE=$(gunzip -c "$datadir/fbank/baker_zh_cuts_test.jsonl.gz" | wc -l)
  N_TAIL=$((N_VA + N_TE))
  lhotse subset --last  $N_TAIL data/fbank/baker_zh_cuts.jsonl.gz data/fbank/vt.jsonl.gz >/dev/null 2>&1
  lhotse subset --first $N_VA   data/fbank/vt.jsonl.gz data/fbank/baker_zh_cuts_valid.jsonl.gz >/dev/null 2>&1
  lhotse subset --last  $N_TE   data/fbank/vt.jsonl.gz data/fbank/baker_zh_cuts_test.jsonl.gz >/dev/null 2>&1
  lhotse subset --first $N_TR   data/fbank/baker_zh_cuts.jsonl.gz data/fbank/baker_zh_cuts_train.jsonl.gz >/dev/null 2>&1
  rm -f data/fbank/vt.jsonl.gz
  "$PY" local/compute_fbank_statistics.py data/fbank/baker_zh_cuts_train.jsonl.gz data/fbank/cmvn.json \
    >/dev/null 2>&1 || { echo "!! cmvn 失败"; return 1; }
  echo "  cuts=$N_TR/$N_VA/$N_TE  cmvn=$(cat data/fbank/cmvn.json)"
  "$PY" -c "
import json
m=json.load(open('data/fbank/cmvn.json'))
print('  该臂归一化对 fbank_mean=%.4f fbank_std=%.4f（导出反归一化必须同一对）' % (m['fbank_mean'], m['fbank_std']))"
}

check_export() {        # $1=导出的 onnx  $2=cmvn.json —— 归一化/反归一化必须同一对（首轮 92.2% 的病灶）
  "$PY" - "$1" "$2" <<'PYEOF' || exit 1
import json, onnx, sys
import numpy as np
f, cmvn = sys.argv[1], sys.argv[2]
cm = json.load(open(cmvn)); mean, std = cm["fbank_mean"], cm["fbank_std"]
g = onnx.load(f)
md = {k.key: k.value for k in g.metadata_props}
assert md.get("sample_rate") == "16000", md.get("sample_rate")
assert "add_blank" not in md, "端侧见到 add_blank 就会插 blank"
vals = np.concatenate([onnx.numpy_helper.to_array(i).ravel().astype(float)
                       for i in g.graph.initializer
                       if onnx.numpy_helper.to_array(i).dtype.kind == "f"])
hit = (int(np.isclose(vals, mean, atol=1e-4).sum()), int(np.isclose(vals, std, atol=1e-4).sum()))
v1 = (int(np.isclose(vals, -2.9522, atol=1e-4).sum()), int(np.isclose(vals, 5.4468, atol=1e-4).sum()))
print(f"   导出物：sample_rate=16000 ✓ 无 add_blank ✓  "
      f"本臂 mel 对({mean:.4f},{std:.4f})命中 {hit[0]}/{hit[1]}  V1 对命中 {v1[0]}/{v1[1]}")
assert all(hit), "导出物里没有本臂训练用的 mel 常数 → 导出守卫没生效，这一臂作废"
PYEOF
}

train_arm() {           # $1=臂名  $2=语料目录  $3=额外 train.py 参数（故意不加引号做词分割，只传本脚本自己写的值）
  # ⚠ 一行 `local a=.. b=$a` 在 set -u 下会炸（同行前面的赋值在 local 语句里还没生效）→ 分行写
  local arm="$1"
  local corpus="$2"
  local extra="$3"
  local datadir="$REPO/data-ab-$arm"
  local exp="matcha/exp-ab-$arm"
  echo ""
  echo "===== 臂 $arm：$corpus  额外参数「$extra」 ====="
  build_data "$corpus" "$datadir" || { echo "!! 臂 $arm 数据链失败，停"; exit 1; }
  rm -rf "$exp"; mkdir -p "$exp"
  "$PY" "$REPO/scripts/make_warmstart_ckpt.py" --bin "$REPO/models/zh-en/pytorch_model_sherpa.bin" \
    --tokens data/tokens.txt --out "$exp/epoch-1.pt" >/tmp/ab-warm-$arm.log 2>&1 || {
      echo "!! 热启动失败"; tail -8 /tmp/ab-warm-$arm.log; exit 1; }
  # 各臂只有 `$extra` 这一处不同：seed 都用 train.py 默认 42、轮数/采样/md/warmstart 全一样
  "$PY" matcha/train.py --exp-dir "$exp" --start-epoch 2 --num-epochs $((EP + 1)) \
    --max-duration "$MD" --bucketing-sampler 1 --num-workers 4 --num-buckets "$NB" \
    --save-every-n 20 $extra >/tmp/ab-train-$arm.log 2>&1 || {
      echo "!! 训练失败"; tail -15 /tmp/ab-train-$arm.log; exit 1; }
  # 这两行是"参数真的生效了"的证据，不看就等于白跑：--freeze 打错前缀会静默地什么都没冻
  grep -E "Optimizer lr overridden|Number of trainable parameters|Frozen .*parameter tensors" \
    /tmp/ab-train-$arm.log | tail -3
  grep -E "batch 0,|Epoch $((EP + 1))," /tmp/ab-train-$arm.log | tail -2
  local ck="epoch-$((EP + 1)).pt"
  [ -f "$exp/$ck" ] || ck="$(ls "$exp"/epoch-*.pt 2>/dev/null | sed -n 's/.*epoch-\([0-9]*\)\.pt/\1/p' | sort -n | tail -1)"
  [ -n "$ck" ] || { echo "!! 臂 $1 没有 ckpt"; exit 1; }
  local ep_num; ep_num="$(echo "$ck" | tr -dc '0-9')"
  echo "-- 导出臂 $arm（epoch $ep_num；导出物写在 CWD，导完立刻改名）"
  "$PY" matcha/export_onnx.py --exp-dir "$exp" --epoch "$ep_num" \
    --tokens data/tokens.txt --cmvn data/fbank/cmvn.json > "/tmp/ab-export-$arm.log" 2>&1 \
    && mkdir -p cand && cp model-steps-3.onnx "cand/steps-3-ab-$arm-ep$ep_num.onnx" \
    && rm -f model-steps-{2,4,5,6}.onnx \
    && ls -lh "cand/steps-3-ab-$arm-ep$ep_num.onnx" || {
      echo "!! 臂 $1 导出失败"; tail -15 "/tmp/ab-export-$arm.log"; exit 1; }
  if grep -q "overriding with --cmvn" "/tmp/ab-export-$arm.log"; then
    echo "   导出守卫生效：用 --cmvn 覆写了 ckpt 的 mel buffer（两臂各自的 cmvn 不同，这正是必须的）"
  else
    echo "   日志无覆写告警（ckpt buffer 已与 --cmvn 一致）"
  fi
  # 结构自检必须真的被调用（定义完忘记接是 2026-10-25 这轮自己踩的：日志里一直没出现这行）
  cp data/fbank/cmvn.json "cand/cmvn-ab-$arm.json"
  check_export "cand/steps-3-ab-$arm-ep$ep_num.onnx" "cand/cmvn-ab-$arm.json"
}

for spec in "${ARMS[@]}"; do
  arm="$(echo "$spec" | cut -d'|' -f1)"
  cdir="$(echo "$spec" | cut -d'|' -f2)"
  xarg="$(echo "$spec" | cut -d'|' -f3)"
  train_arm "$arm" "$cdir" "$xarg"
done

echo ""
echo "[ab] 产物："
ls -lh cand/steps-3-ab-*.onnx cand/cmvn-ab-*.json 2>/dev/null
tar czf /root/ab_onnx.tgz cand/steps-3-ab-*.onnx cand/cmvn-ab-*.json 2>/dev/null \
  || tar czf /root/ab_onnx.tgz cand/steps-3-ab-*.onnx
ls -lh /root/ab_onnx.tgz
cat <<'EOF'
[ab] 判读（判据在跑之前就登记在这里，不许事后挪）：
  主对照始终是 V1 自己：held-out 音口径 24.3%、语音时长比 1.000、每条固定静音 ~0.75s。
  ⚠ 时长一律看 `duration_gate.py` 报的**语音时长比**——整段时长含固定首尾静音，
     拿它当语速已经错过一次（A/B 那轮的机制解释就是这么错的，见 flow.md §7.2）。
  ① 不伤害门（三臂都要过）：held-out 音口径 ≤ 30%（V1 + 5pp，这 5pp 是本尺子的重采样噪声带），
     且语音时长中位比 ∈[0.95,1.05]、无单条 <0.85。
  ② 收益门（过了①才有资格谈）：`时间`（V1 32%）或`多音字`（V1 45%）任一类改善 ≥10pp。
     注意 A/B 那两臂是"零条改善 >5pp"，所以这一条才是真正的门槛。
  ③ 三臂都过不了① → 0.92h 跨引擎教师蒸馏这条路判死，转去谈调用侧文本改写或先扩语料量级再重估。
     d 过①而 c 不过 → replay 是必需项，以后每轮都带。
     e 明显优于 c → 遗忘主要在 CFM 主干，后续锁 estimator 只调 encoder/dur。
[ab] 本地判读命令（同一台 Mac，与 V1 基线同机才可比）：
  scp -p <实例>:/root/ab_onnx.tgz /tmp/ && tar xzf /tmp/ab_onnx.tgz -C out/ab
  for a in c d e; do
    d=out/ab/$a; mkdir -p $d; ln -sf $PWD/models/zh-en/vocos-16khz-univ.onnx $d/
    cp out/ab/cand/steps-3-ab-$a-ep*.onnx $d/model-steps-3.onnx
    .venv/bin/python scripts/duration_gate.py --cand $d --n 4
    TTS_MODEL_DIR=$PWD/$d .venv/bin/python scripts/ruler.py --n 30 --threads 4 \
      --phrases-file scripts/ruler_heldout.tsv | tail -1
  done
EOF
