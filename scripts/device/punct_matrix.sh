#!/bin/sh
# 设备侧：标点存活矩阵。同一条话术的多种边界写法各合成一次，报 token 行数 / 推理次数 / 时长。
#
# 为什么要在设备上跑而不是在 PC 复刻前端跑：本仓的 PC 复刻（scripts/g2p.py）不拆段，
# 而设备上的 sherpa 前端遇到 IsPunct 会把一条播报切成多次独立推理——"加标点能不能降错率"
# 这个结论只有在设备上才成立（见 docs/device-runtime-check.md）。
#
# 这个脚本跑在设备上（由 pull_and_manifest.sh 那一侧用 hdc 推上去再执行），所以：
#   - 只用 POSIX sh：设备上没有 expr，`while [ $I -le N ]` + expr 会静默只跑一轮；
#   - 路径全部相对脚本自身目录，模型目录用 $M 覆盖；
#   - 绝不写生产目录（rawfile / filesDir），只在 $TTS_WORK 里落 wav 和日志。
#
# 用法（设备上）：
#   sh punct_matrix.sh                       # 读同目录 cases.tsv，模型在 ./matcha
#   M=zh-en CASES=my_cases.tsv sh punct_matrix.sh
# 输入 cases.tsv：每行 `id <TAB> 文本`，文本里可写 \n（脚本用 printf '%b' 展开成真换行，
# 用来验前端把 \s+ 归一成空格的行为）。
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
M="${M:-matcha}"
CASES="${CASES:-cases.tsv}"
mkdir -p out logs

IFS='
'
for LINE in `cat "$CASES"`; do
  ID=`printf '%s' "$LINE" | sed 's/	.*//'`
  case "$ID" in \#*|"") continue ;; esac
  TXT=`printf '%s' "$LINE" | sed 's/^[^	]*	//'`
  TXT=`printf '%b' "$TXT"`

  # 参数与生产一致：noise_scale=1.0 + 三个 -zh rule-fst + 4 线程。
  # 注意 TTS 的开关叫 --tts-rule-fsts；--rule-fsts 是 ASR 的，传错会走"未知选项"分支 exit(-1)，
  # 也就是 RC=255，很容易被误判成"这颗模型带 fst 会崩"。
  set -- \
    --matcha-acoustic-model="$M/model-steps-3.onnx" \
    --matcha-vocoder="$M/vocos-16khz-univ.onnx" \
    --matcha-lexicon="$M/lexicon.txt" \
    --matcha-tokens="$M/tokens.txt" \
    --matcha-data-dir="$M/espeak-ng-data" \
    --matcha-noise-scale="${NOISE_SCALE:-1.0}" \
    --tts-rule-fsts="$M/number-zh.fst,$M/date-zh.fst,$M/phone-zh.fst" \
    --num-threads="${THREADS:-4}" --debug=1

  hilog -r > /dev/null 2>&1
  ./bin/sherpa-onnx-offline-tts "$@" --output-filename="out/$ID.wav" "$TXT" > "logs/$ID.out" 2> "logs/$ID.err"
  # token 级证据只在 hilog 里；CLI 的 stderr 只有 config 回显。
  hilog -x 2> /dev/null | grep 'sherpa_onnx' | sed 's/.*sherpa_onnx: //' > "logs/$ID.hi"

  CB=`grep -c 'progress=' "logs/$ID.out"`
  NS=`grep -o 'Number of sentences: [0-9]*' "logs/$ID.hi" | tail -1 | sed 's/.*: //'`
  TK=`grep -cE "'\\([0-9]+\\)," "logs/$ID.hi"`
  DUR=`grep -o 'Audio duration: [0-9.]*' "logs/$ID.err" | tail -1 | sed 's/.*: //'`
  ELAP=`grep -o 'Elapsed seconds: [0-9.]*' "logs/$ID.err" | tail -1 | sed 's/.*: //'`
  echo "ID=$ID | callbacks=$CB | sentences=${NS:-1} | token_lines=$TK | audio_dur=$DUR | elapsed=$ELAP"
done
echo MATRIX_DONE
