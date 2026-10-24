#!/usr/bin/env bash
# 建 venv + 装依赖（含 piper_phonemize）。国内 github 直连常 403，走 gh-proxy。
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$REPO/.venv}"
PYTHON="${PYTHON:-python3}"   # 需要 >=3.10；例：PYTHON=/opt/homebrew/bin/python3.10 bash setup.sh

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r "$REPO/requirements.txt"

# piper_phonemize：PyPI 无常规包，从 github release 下 wheel（按 python 版本+平台自动探测）
# 必须保留完整 wheel 文件名再 pip 安装，否则报 not a valid wheel filename
TAG="$("$VENV/bin/python" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
PLAT="$("$VENV/bin/python" -c 'import sys,platform; s=platform.system().lower(),platform.machine().lower(); print({("darwin","arm64"):"macosx_11_0_arm64",("linux","x86_64"):"manylinux2014_x86_64.manylinux_2_17_x86_64",("linux","aarch64"):"manylinux2014_aarch64.manylinux_2_17_aarch64"}.get(s,"unknown"))')"
WHL="piper_phonemize-1.5.0-${TAG}-${TAG}-${PLAT}.whl"
if [ "$PLAT" = "unknown" ]; then
  echo "[piper] 平台未识别（$PLAT），跳过 wheel 下载"
else
  echo "[piper] 下载 $WHL （失败就手动从 https://github.com/csukuangfj/piper-phonemize/releases 取对应 wheel）"
  curl -L --max-time 180 -s "https://gh-proxy.com/https://github.com/csukuangfj/piper-phonemize/releases/download/v1.5.0/$WHL" \
    -o "/tmp/$WHL" && "$VENV/bin/python" -m pip install -q "/tmp/$WHL" || \
    echo "[warn] piper_phonemize 未装；英文 G2P 会退回 espeak-ng（需 apt install espeak-ng）"
fi

echo "完成。激活：source $VENV/bin/activate"
echo "下一步：把 V1 模型放 models/zh-en/、SenseVoice 放 models/sense-voice/、vocab_tts.txt 放 models/（见 README）"
