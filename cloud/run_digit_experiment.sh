#!/usr/bin/env bash
# 定向数字实验的一轮总入口（一次开机跑完）：老师合成 → 混合语料 → 训两臂 → 导出。
#
# 为什么要这一发（2026-10-25，见 docs/icefall-finetune-flow.md §7.4）：
#   ① 收益尺 ruler_digits 实测 V1 在**没见过的数字取值**上就 24.1% 错，其中三位床号 47%、
#      "五点五十分"这类是**复制**型错（多念一遍相邻音节）——这是微调治得了的病；
#   ② 旧语料把 1..99 床号读法吃到只剩 2 个空白位，所以含数字的旧句必须整批剔出训练集，
#      否则"泛化"读数全是假的（mix_corpus 的 @nodigit 就是为这件事加的）；
#   ③ 前端读错的那类（大夫、儿化）不在本轮射程内，微调治不了。
#
# 前置（实例上，全部走仓里已审脚本）：
#   . /root/env_gpu.sh && cd $REPO && git pull --ff-only
#   bash cloud/bootstrap_gpu.sh && bash cloud/smoke.sh      # 新实例/换厂商才需要
# 参考音色素材不在仓里（大文件），开机后从 Mac 传一次：
#   scp out/ref_v1.wav out/ref_v1.txt <实例>:/root/matcha-tts-finetune/
#   scp -r out/corpus_nodigit <实例>:/root/matcha-tts-finetune/out/   # 42M，409 条无数字句
# 无数字子集在 Mac 上这样切（旧语料含数字的 762 句必须剔掉，否则数字留出作废）：
#   .venv/bin/python scripts/mix_corpus.py --out out/corpus_nodigit \
#       out/corpus1k@nodigit out/replay_v1@nodigit
#
# 用法（实例上）：
#   . /root/env_gpu.sh
#   nohup bash $REPO/cloud/run_digit_experiment.sh > /tmp/digit.log 2>&1 &
#   tail -f /tmp/digit.log
# 时长预算：合成 600 句 ≈60 分钟（--n 8 拒绝采样实测 ≈6 秒/句，大头在重试），
# 混合 <1 分钟，训练两臂 ≤10 分钟（A/B 实测一臂含建数据 ≤4 分钟），合计 ≈75 分钟 ≈¥2.7。
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
COSY="${COSY:-$HOME/CosyVoice}"
PY="${PY:-python3}"
SCRIPT="$REPO/data/record_script_digits.tsv"
EP="${AB_EPOCHS:-40}"
N="${DIGIT_N:-8}"                      # 每条最多重试次数（n=8 时实测老师首试过 75%）
REF_WAV="${REF_WAV:-$REPO/ref_v1.wav}"
REF_TXT="${REF_TXT:-$REPO/ref_v1.txt}"
MODEL_DIR="${COSY_MODEL:-$COSY/pretrained_models/iic/CosyVoice2-0.5B}"

say() { echo ""; echo "===== $* ====="; }

[ -s "$SCRIPT" ] || { echo "!! 缺 $SCRIPT，先在 Mac 上跑 scripts/make_digit_script.py 并 git push"; exit 1; }
[ -f "$REF_WAV" ] || { echo "!! 缺参考音色 $REF_WAV（从 Mac scp 上来，见脚本头）"; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "!! 缺老师权重 $MODEL_DIR（跑 bootstrap_gpu.sh §5 下载）"; exit 1; }
NLINE=$(grep -cv '^#' "$SCRIPT")
echo "[digit] 定向话术 $NLINE 条；老师 $MODEL_DIR；n=$N"

say "[1/3] 老师合成（CPU 门控在 SenseVoice 上跑，ASR 不过的样本进 rejected/）"
cd "$REPO" || exit 1
PYTHONPATH="$COSY" "$PY" scripts/gen_teacher_corpus.py --backend cosyvoice \
  --model-dir "$MODEL_DIR" --ref-wav "$REF_WAV" --ref-text "$(cat "$REF_TXT")" \
  --script "$SCRIPT" --n "$N" --out out/corpus_digits || { echo "!! 合成失败"; exit 1; }
HAVE=$(ls out/corpus_digits/wav 2>/dev/null | wc -l | tr -d ' ')
echo "[digit] 合成产出 $HAVE/$NLINE 句（report.log 有错率）"
# 老师产出率门：低于 60% 说明这批句式老师自己也念不好，训了也是灌错
if [ "$HAVE" -lt $(( NLINE * 6 / 10 )) ]; then
  echo "!! 老师产出率 $HAVE/$NLINE < 60%，先别训——看 out/corpus_digits/rejected/ 人耳复核"
  exit 1
fi

say "[2/3] 混合训练集：定向数字 + 无数字旧句/回放（409 条，已在 Mac 上切好子集）"
[ -d out/corpus_nodigit/wav ] || { echo "!! 缺 out/corpus_nodigit —— 先从 Mac 传（见脚本头 scp）"; exit 1; }
"$PY" scripts/mix_corpus.py --out out/corpus_digit \
  out/corpus_digits out/corpus_nodigit \
  || { echo "!! 混合失败"; exit 1; }
wc -l out/corpus_digit/transcripts.tsv

say "[3/3] 训两臂并导出（判据已登记在 cloud/ab_rate.sh 尾部）"
bash "$REPO/cloud/ab_rate.sh" || { echo "!! 训练/导出失败"; exit 1; }

cat <<EOF

[digit] 跑完了。产物在实例上：
  $REPO/out/corpus_digit/            混合语料（wav + transcripts.tsv）
  \$TTS/cand/steps-3-ab-{f,g}-ep*.onnx + cmvn-ab-*.json，打包在 /root/ab_onnx.tgz
回 Mac 判读（与 V1 基线同机才可比）：
  mkdir -p out/ab && scp -p <实例>:/root/ab_onnx.tgz /tmp/ && tar xzf /tmp/ab_onnx.tgz -C out/ab
  然后按 ab_rate.sh 尾部那段 for a in f g 的循环跑 duration_gate + 两把尺子。
EOF
