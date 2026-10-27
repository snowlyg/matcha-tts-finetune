#!/usr/bin/env bash
# 检查一份 sherpa-onnx OHOS 预编译包**到底能不能给这台设备用**（浮点 ABI + SONAME 依赖）。
#
# 为什么要有：判"上游没出这个架构的包"和"出的包是硬浮所以加载不了"是两件不同的事，
# 而失败信息都只是一行 `dlopen ... symbol not found`。跑一遍这个脚本就能分清。
#
# 用法：SHERPA_PKG=/path/to/sherpa-onnx-v2.0.0-ohos-armeabi-v7a bash scripts/zipvoice/check_abi.sh
# 期望（32 位软浮设备）：Tag_CPU_arch=v7、Tag_FP_arch=VFPv3、**没有** Tag_ABI_VFP_args=VFP registers
set -uo pipefail
PKG="${SHERPA_PKG:?需要 SHERPA_PKG=<解压后的 ohos 包目录>}"
LIBDIR="$PKG/lib"
[ -d "$LIBDIR" ] || { echo "!! 没有 $LIBDIR"; exit 1; }

for f in libsherpa-onnx-c-api.so libonnxruntime.so libsherpa-onnx-cxx-api.so; do
  p="$LIBDIR/$f"
  echo "### $f"
  [ -f "$p" ] || { echo "   缺失"; continue; }
  file "$p" | cut -c1-160
  readelf -A "$p" 2>/dev/null | grep -iE "CPU_name|CPU_architecture|FP_arch|VFP_args" | sed 's/^/   /'
  readelf -d "$p" 2>/dev/null | grep -iE "SONAME|NEEDED" | sed 's/^/   /' | head -8
done

echo
echo "判读：32 位软浮设备只认 'ELF 32-bit ARM' + Tag_FP_arch=VFPv3 且无 Tag_ABI_VFP_args；"
echo "      armeabi-v7a 的包如果是 hard-float，设备上就是 dlopen 失败，不是模型问题。"
