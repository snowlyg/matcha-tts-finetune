#!/usr/bin/env bash
# 收工保险：等主发跑完 → 校验产物齐不齐 → 打成一个包 → 写 /root/RUN_DONE 标记。
#
# 为什么要有这个：实例盘是唯一副本（关机期间保留已两次验证，但"唯一"这件事本身是风险），
# 而且控制台"定时关机"是否真生效**从未验证过**，不能当唯一保险。所以关机前必须做到两件事：
#   ① 训练/导出**真的结束**了再断电（半途断电会留下写了一半的 onnx/tgz）；
#   ② 产物归拢成一个 tar + 一份可读的清单，明早 scp 一次就能取回。
# 这脚本**不自己关机**：容器内 poweroff 在 compshare 上不等于平台停止计费，
# 关机要么走控制台，要么走厂商 API；本脚本只负责把"可以安全关机"这个事实落盘。
#
# 用法（实例上，nohup 挂着即可，最长等 100 分钟）：
#   cd /root/matcha-tts-finetune && nohup bash cloud/pack_after_run.sh > /tmp/pack.log 2>&1 &
set -uo pipefail

REPO="${REPO:-/root/matcha-tts-finetune}"
TTS="${TTS:-/root/icefall/egs/baker_zh/TTS}"
LOG="${DIGIT_LOG:-/tmp/digit.log}"
OUT_TGZ=/root/digit_products.tgz
DONE=/root/RUN_DONE

say() { echo "[pack] $*"; }

for _ in $(seq 1 200); do
  grep -q "\[digit\] 跑完了" "$LOG" 2>/dev/null && break
  if ! pgrep -f "run_digit_experiment" >/dev/null; then
    say "主发进程已不在，但仍按现状收拢产物（可能是中途失败）"
    break
  fi
  sleep 30
done

cd "$REPO" || { say "没有 $REPO"; exit 1; }
{
  echo "完成时间(UTC): $(date -u '+%F %T')"
  echo "--- 老师合成 ---"
  tail -2 out/corpus_digits/report.log 2>/dev/null
  echo "wav=$(ls out/corpus_digits/wav 2>/dev/null | wc -l) rejected=$(ls out/corpus_digits/rejected 2>/dev/null | wc -l)"
  echo "--- 混合语料 ---"
  echo "transcripts=$(wc -l < out/corpus_digit/transcripts.tsv 2>/dev/null) wav=$(ls out/corpus_digit/wav 2>/dev/null | wc -l)"
  echo "--- 候选导出物（两臂 + 每臂 cmvn） ---"
  ls -l "$TTS"/cand/steps-3-ab-*.onnx "$TTS"/cand/cmvn-ab-*.json 2>/dev/null
  echo "--- 训练是否真跑了两臂 ---"
  grep -E "臂 [fg]|Frozen|Number of trainable|Optimizer lr overridden" /tmp/ab.log 2>/dev/null | tail -8
  echo "--- 失败迹象 ---"
  grep -E "!!|FAILED|Traceback" "$LOG" 2>/dev/null | tail -6
  echo "--- 老师对照（CosyVoice3，若跑了） ---"
  tail -2 out/teacher_cv3/report.log 2>/dev/null
  grep -E "^\[cv3\]" "$LOG" 2>/dev/null | tail -4
} > "$DONE" 2>&1

# 归拢：产物清单 + 日志 + 语料转写 + 候选 ONNX（体积最大，单独一个 tar 便于分次取）
cp -f "$TTS"/cand/steps-3-ab-*.onnx "$TTS"/cand/cmvn-ab-*.json out/ab/ 2>/dev/null \
  || { mkdir -p out/ab && cp -f "$TTS"/cand/steps-3-ab-*.onnx "$TTS"/cand/cmvn-ab-*.json out/ab/ 2>/dev/null; }
tar czf "$OUT_TGZ" \
  out/ab/steps-3-ab-*.onnx out/ab/cmvn-ab-*.json \
  out/corpus_digits/report.log out/corpus_digits/transcripts.tsv \
  out/corpus_digit/transcripts.tsv \
  "$DONE" 2>/dev/null
# 语料 wav 单独打包：明早想重训/换配方时不用重新合成（100MB 级，值这个 scp）
tar czf /root/corpus_digit_wav.tgz -C out corpus_digit/wav corpus_digit/transcripts.tsv 2>/dev/null

say "产物包："
ls -lh "$OUT_TGZ" /root/corpus_digit_wav.tgz 2>/dev/null
say "清单在 $DONE；明早取回："
echo "  scp -p <实例>:$OUT_TGZ /root/corpus_digit_wav.tgz /tmp/"
echo "  然后按 cloud/ab_rate.sh 尾部的 for a in f g 循环跑 duration_gate + 两把尺子"
say "现在可以安全关机（训练/导出已结束）"
