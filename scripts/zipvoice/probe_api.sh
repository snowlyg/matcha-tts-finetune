#!/usr/bin/env bash
# 取某个上游 tag 的 C API 头 + ZipVoice 的 c-api 示例，把结构体字段和生成接口打出来。
#
# 为什么要它：换基座前必须先确认**现装的 runtime 吃不吃得下这个包**。ZipVoice 的字段
# （encoder/decoder/data_dir/feat_scale/t_shift/target_rms/guidance_scale）和
# `SherpaOnnxOfflineTtsGenerateWithConfig` 是随版本走的，光看 release 页说不清，
# 直接 diff 头文件最快。
#
# 用法：SHERPA_TAG=v2.0.0 bash scripts/zipvoice/probe_api.sh
# 代理/镜像自己换 RAW_BASE；判定哪个 tag 才支持某个特性，用上游的
# `git tag --contains <commit>`，别拿 release 说明当准。
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
TAG="${SHERPA_TAG:-v2.0.0}"
RAW_BASE="${RAW_BASE:-https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/$TAG}"
WORK="${ZV_WORK:-$REPO/out/zipvoice-build}"
mkdir -p "$WORK/api"

for pair in "sherpa-onnx/c-api/c-api.h:c-api.h" "c-api-examples/zipvoice-tts-zh-en-c-api.c:zipvoice_example.c"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  wget -q -c "$RAW_BASE/$src" -O "$WORK/api/$dst" || echo "!! 取 $src 失败（tag=$TAG）"
done

H="$WORK/api/c-api.h"
[ -s "$H" ] || { echo "!! 没有 c-api.h"; exit 1; }
echo "c-api.h bytes=$(stat -c%s "$H")  tag=$TAG"

echo "=== ZipVoice 结构体字段 ==="
grep -n "Zipvoice" "$H" | head -24
echo "=== 生成接口与参考音频字段 ==="
grep -n "GenerateWithConfig\|GenerationConfig\|reference_audio\|reference_text\|num_steps" "$H" | head -24
echo "=== 官方示例怎么填这些字段（C API 结构体没有默认值，示例是唯一口径）==="
grep -nE "feat_scale|t_shift|target_rms|guidance_scale|data_dir|lexicon|num_steps" \
  "$WORK/api/zipvoice_example.c" 2>/dev/null | head -24
