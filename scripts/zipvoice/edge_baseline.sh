#!/bin/sh
# 设备侧同机分母：现网这颗 Matcha zh-en(V1) 在**同一台设备、同一线程数**下逐句的
# elapsed / 音频时长 / RTF / 峰值 RSS。RTF 外推必须先有这个，不然"能不能跑"只能靠猜。
#
# 为什么要峰值 RSS 而不只看时长：换基座的真实门槛常常是内存而不是速度。
# 这台设备的 shell 没有 awk，`/proc/<pid>/status` 的 VmHWM 用 sed 抠；
# 峰值要在进程跑的过程中轮询，跑完再读就被回收了。
#
# 用法（设备上，脚本与 bin/、lib/、模型目录同级）：
#   sh edge_baseline.sh                      # 模型在 ./matcha，话术在同目录 phrases.txt
#   M=zh-en THREADS=4 SHARDS=20 sh edge_baseline.sh
# 与生产一致：noise_scale=1.0 + 三个 -zh rule-fst + threads=4。
# 红线：只在自己的工作目录里写，不碰生产的 rawfile / filesDir。
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
M="${M:-matcha}"
PHRASES="${PHRASES:-phrases.txt}"

set -- \
  --matcha-acoustic-model="$M/model-steps-3.onnx" \
  --matcha-vocoder="$M/vocos-16khz-univ.onnx" \
  --matcha-lexicon="$M/lexicon.txt" \
  --matcha-tokens="$M/tokens.txt" \
  --matcha-data-dir="$M/espeak-ng-data" \
  --matcha-noise-scale="${NOISE_SCALE:-1.0}" \
  --tts-rule-fsts="$M/number-zh.fst,$M/date-zh.fst,$M/phone-zh.fst" \
  --num-threads="${THREADS:-4}"

i=0
IFS='
'
for T in `cat "$PHRASES"`; do
  case "$T" in \#*|"") continue ;; esac
  i=$((i + 1))
  ./bin/sherpa-onnx-offline-tts "$@" --output-filename=_base.wav "$T" > /dev/null 2> _err.log &
  P=$!
  PEAK=0
  while kill -0 $P 2> /dev/null; do
    L=`sed -n 's/^VmHWM:[ ]*\([0-9][0-9]*\).*/\1/p' /proc/$P/status 2> /dev/null`
    [ "X$L" = "X" ] || [ "$L" -le "$PEAK" ] || PEAK=$L
    sleep 0.2
  done
  wait $P
  E=`sed -n 's/^Elapsed seconds: //p' _err.log`
  D=`sed -n 's/^Audio duration: //p' _err.log`
  R=`sed -n 's/.*RTF): //p' _err.log`
  echo "CASE=$i | elapsed=$E dur=$D rtf=[$R] | peak_kb=$PEAK | text=$T"
done
echo "BASELINE_DONE cases=$i"
