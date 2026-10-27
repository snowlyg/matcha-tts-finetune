#!/bin/sh
# 设备侧：同句 N 次 A/B 复测。每一轮把同一批条件各合成一次（**交错**，不是先跑完 A 再跑 B），
# 这样设备温度/负载漂移对两档的影响是一样的。
#
# 为什么必须重复：这颗模型是扩散采样，同句同参数每次结果都不同，单次回读不算结论
# （docs/icefall-finetune-flow.md 里那些百分比都是 n≥8 的复现率）。n=8 只够判
# "有没有新增错误类型"，不够报百分比。
#
# 用法（设备上）：
#   sh ab_repeat.sh                              # 读同目录 cases.tsv，每档 8 次
#   REPS_LIST='1 2 3 4' CASES=pairs.tsv sh ab_repeat.sh
# 输入格式同 punct_matrix.sh：每行 `id <TAB> 文本`；约定成对的 id 用同一前缀 + a/b 后缀
# （如 `E_a` / `E_b`），measure_pause.py 与 score_wavs.py 的清单都按这个约定配对。
# 产物：rep/<id>_r<N>.wav
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
M="${M:-matcha}"
CASES="${CASES:-cases.tsv}"
mkdir -p rep

set -- \
  --matcha-acoustic-model="$M/model-steps-3.onnx" \
  --matcha-vocoder="$M/vocos-16khz-univ.onnx" \
  --matcha-lexicon="$M/lexicon.txt" \
  --matcha-tokens="$M/tokens.txt" \
  --matcha-data-dir="$M/espeak-ng-data" \
  --matcha-noise-scale="${NOISE_SCALE:-1.0}" \
  --tts-rule-fsts="$M/number-zh.fst,$M/date-zh.fst,$M/phone-zh.fst" \
  --num-threads="${THREADS:-4}"

for R in ${REPS_LIST:-1 2 3 4 5 6 7 8}; do
  IFS='
'
  for LINE in `cat "$CASES"`; do
    ID=`printf '%s' "$LINE" | sed 's/	.*//'`
    case "$ID" in \#*|"") continue ;; esac
    TXT=`printf '%s' "$LINE" | sed 's/^[^	]*	//'`
    TXT=`printf '%b' "$TXT"`
    ./bin/sherpa-onnx-offline-tts "$@" --output-filename="rep/${ID}_r${R}.wav" "$TXT" > /dev/null 2>&1
  done
  echo "REP=$R done"
done
echo AB_DONE
