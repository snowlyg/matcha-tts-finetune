#!/usr/bin/env bash
# PC 侧总控：把设备脚本和话术表推上去、在设备上跑、把 wav 拉回 out/device/<batch>/，
# 并顺手生成 score_wavs.py 要的清单（TSV：期望读音 <TAB> wav 路径 <TAB> 类别）。
#
# 为什么要有清单这一步：喂进去的文本和**该念出来的读音**不是一回事——`21床。正在呼叫`
# 喂的是数字加句号，回读期望是 `二十一床正在呼叫`（fst 展开、标点不发音）。打分必须按读音比，
# 所以话术表里 fed 与 expected 分两列，别拿 fed 当期望。
#
# 环境（都不写死，换设备只改这里）：
#   TTS_TARGET   hdc 目标，形如 <设备IP>:5555（`hdc list targets` 可列出）；只连一台时可留空
#   TTS_DEV_DIR  设备工作目录，默认 /data/local/tmp/tts-probe（**不要指向生产 rawfile/filesDir**）
#   TTS_BATCH    本地批次名，落 out/device/<batch>/（out/ 不入库）
#   MODE         matrix=标点存活矩阵（每档一次） | ab=同句 N 次 A/B（交错重复）
#   CASES        本地话术表，默认 data/punct_cases.tsv（四列：id / fed / expected / group）
#   REPS_LIST    ab 模式的重复轮次，默认 '1 2 3 4 5 6 7 8'
#
# 前置（一次性，脚本不代做）：设备 $TTS_DEV_DIR 下要已有 bin/sherpa-onnx-offline-tts、lib/、
# 以及 $M 模型目录（model-steps-3.onnx / vocos / lexicon.txt / tokens.txt / espeak-ng-data /
# 三个 -zh fst）。模型不入库，来源见 docs/edge-deploy-constraints.md。
#
# 用法：
#   TTS_TARGET=... MODE=matrix bash scripts/device/pull_and_manifest.sh
#   TTS_TARGET=... MODE=ab CASES=data/ab_bed.tsv bash scripts/device/pull_and_manifest.sh
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
TARGET="${TTS_TARGET:-}"
DEV="${TTS_DEV_DIR:-/data/local/tmp/tts-probe}"
BATCH="${TTS_BATCH:-punct}"
MODE="${MODE:-matrix}"
CASES="${CASES:-$REPO/data/punct_cases.tsv}"
REPS="${REPS_LIST:-1 2 3 4 5 6 7 8}"
LOCAL="$REPO/out/device/$BATCH"

HDC=(hdc)
[ -n "$TARGET" ] && HDC=(hdc -t "$TARGET")

run() { "${HDC[@]}" shell "$1"; }
send() { "${HDC[@]}" file send "$1" "$2" > /dev/null; }

[ -f "$CASES" ] || { echo "!! 话术表不存在: $CASES"; exit 1; }
mkdir -p "$LOCAL"
run "mkdir -p '$DEV'" > /dev/null

# 设备侧只吃两列（id / fed），第三四列是 PC 侧打分用的
cut -f1,2 "$CASES" | tr -d '\r' > "$LOCAL/cases.tsv"
send "$LOCAL/cases.tsv" "$DEV/cases.tsv"

case "$MODE" in
  matrix)
    for f in punct_matrix.sh; do
      tr -d '\r' < "$REPO/scripts/device/$f" > "$LOCAL/$f"; send "$LOCAL/$f" "$DEV/$f"
    done
    run "cd '$DEV' && sh punct_matrix.sh"
    SUB=out
    ;;
  ab)
    for f in ab_repeat.sh; do
      tr -d '\r' < "$REPO/scripts/device/$f" > "$LOCAL/$f"; send "$LOCAL/$f" "$DEV/$f"
    done
    run "cd '$DEV' && REPS_LIST='$REPS' sh ab_repeat.sh"
    SUB=rep
    ;;
  *)
    echo "!! MODE 只能是 matrix 或 ab"; exit 1
    ;;
esac

# hdc file send/recv 给一个不存在的目标名会**建成目录**（踩过三次），所以先确认目标目录存在再拉
LIST=$(run "ls '$DEV/$SUB'" | tr -d '\r' | grep '\.wav$' || true)
mkdir -p "$LOCAL/wav"
N=0
for f in $LIST; do
  "${HDC[@]}" file recv "$DEV/$SUB/$f" "$LOCAL/wav/$f" > /dev/null 2>&1 && N=$((N + 1))
done
echo "pulled=$N -> $LOCAL/wav"

# 清单：期望读音 / wav 路径 / 类别。ab 模式下同一 id 的多个 rep 各占一行。
MAN="$LOCAL/manifest.tsv"
: > "$MAN"
while IFS=$'\t' read -r id fed expected group; do
  case "$id" in \#*|"") continue ;; esac
  if [ "$MODE" = ab ]; then
    for r in $REPS; do
      w="$LOCAL/wav/${id}_r${r}.wav"
      [ -f "$w" ] && printf '%s\t%s\t%s\n' "$expected" "$w" "$group" >> "$MAN"
    done
  else
    w="$LOCAL/wav/${id}.wav"
    [ -f "$w" ] && printf '%s\t%s\t%s\n' "$expected" "$w" "$group" >> "$MAN"
  fi
done < <(tr -d '\r' < "$CASES")
echo "manifest=$(wc -l < "$MAN") -> $MAN"
echo "下一步: python scripts/score_wavs.py --manifest $MAN"
echo "        python scripts/device/measure_pause.py --dir $LOCAL/wav"
