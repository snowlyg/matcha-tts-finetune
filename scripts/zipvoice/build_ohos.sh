#!/usr/bin/env bash
# 交叉编译端侧基准驱动（OHOS / armv7 **softfp**），链官方预编译的 sherpa-onnx C API。
#
# 为什么单独留这个脚本：这台设备的应用二进制必须是 softfp —— 用 `readelf -A` 看必须是
# `Tag_FP_arch: VFPv3` 且**没有** `Tag_ABI_VFP_args`，跑错浮点 ABI 的表现不是报"浮点异常"，
# 而是 dlopen 阶段就符号找不到，很容易被当成"上游没出 OHOS 预编译包"。
# 注意 Windows 侧 SDK 里没有 ELF clang，交叉编译只能在 Linux 上做（官方 OHOS 预编译包见
# docs/edge-deploy-constraints.md）。
#
# 用法（Linux 侧，路径全用环境变量给，不写死本机布局）：
#   OHOS_LLVM=/path/to/llvm/bin OHOS_SYSROOT=/path/to/sysroot \
#   SHERPA_PKG=/path/to/sherpa-onnx-v2.0.0-ohos-armeabi-v7a \
#   bash scripts/zipvoice/build_ohos.sh
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
LLVM="${OHOS_LLVM:?需要 OHOS_LLVM=.../llvm/bin}"
SYSROOT="${OHOS_SYSROOT:?需要 OHOS_SYSROOT=.../sysroot}"
PKG="${SHERPA_PKG:?需要 SHERPA_PKG=.../sherpa-onnx-v2.0.0-ohos-armeabi-v7a}"
TAG="${SHERPA_TAG:-v2.0.0}"
ARCH="${SHERPA_ARCH:-armeabi-v7a}"

WORK="${ZV_WORK:-$REPO/out/zipvoice-build}"
INC="$WORK/inc"
SRC="$REPO/scripts/zipvoice/zipvoice_bench.c"
mkdir -p "$INC/sherpa-onnx/c-api" "$WORK/bin"

# v2.0.0 移除了 espeak-ng/piper-phonemize（#4055），英文读音只能靠 lexicon-en-us.txt
LEX_EN="$WORK/lexicon-en-us.txt"
if [ ! -s "$LEX_EN" ]; then
  wget -q -c "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt" \
    -O "$LEX_EN" || { echo "!! 下载 lexicon-en-us.txt 失败（换网络时自己指个镜像）"; exit 1; }
fi
echo "lexicon-en-us.txt bytes=$(stat -c%s "$LEX_EN")"

# C API 头必须是**目标版本**的那份：结构体字段随版本变，用错版本会 Validate 失败
if [ ! -s "$INC/sherpa-onnx/c-api/c-api.h" ]; then
  wget -q -c "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/$TAG/sherpa-onnx/c-api/c-api.h" \
    -O "$INC/sherpa-onnx/c-api/c-api.h" || { echo "!! 取 c-api.h 失败，tag=$TAG"; exit 1; }
fi

"$LLVM/clang" --target=armv7-unknown-linux-ohos \
  --sysroot="$SYSROOT" \
  -mfloat-abi=softfp -mfpu=neon-vfpv4 -march=armv7-a \
  -O2 -std=c11 \
  -I"$INC" \
  "$SRC" \
  -L"$PKG/lib" -lsherpa-onnx-c-api \
  -Wl,-rpath,'$ORIGIN/lib' \
  -o "$WORK/bin/zipvoice_bench"

echo "=== 产物自检（这三行就是判 ABI 对不对的依据）==="
file "$WORK/bin/zipvoice_bench" | cut -c1-160
readelf -A "$WORK/bin/zipvoice_bench" | grep -iE "FP_arch|VFP_args|CPU_name"
readelf -d "$WORK/bin/zipvoice_bench" | grep -iE "NEEDED|RPATH|RUNPATH"
echo "STAGED: $WORK/bin/zipvoice_bench"
echo "推到设备的 bin/ 后即可跑：./zipvoice_bench <model_dir> <vocos_24k> $LEX_EN <phrases> 4 4"
