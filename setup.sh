#!/usr/bin/env bash
# 建 venv + 装依赖（含 piper_phonemize）。国内 github 直连常 403，走 gh-proxy。
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$REPO/.venv}"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -r "$REPO/requirements.txt"

# piper_phonemize：PyPI 无常规包，从 github release 下 wheel（按你的 python 版本改 cpXXX）
TAG="cp312"   # python3.12 -> cp312；3.10 -> cp310
WHL="piper_phonemize-1.5.0-${TAG}-${TAG}-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
echo "[piper] 下载 $WHL （失败就手动从 https://github.com/csukuangfj/piper-phonemize/releases 取对应 wheel）"
curl -L --max-time 180 -s "https://gh-proxy.com/https://github.com/csukuangfj/piper-phonemize/releases/download/v1.5.0/$WHL" \
  -o "/tmp/$WHL" && "$VENV/bin/python" -m pip install -q "/tmp/$WHL" || \
  echo "[warn] piper_phonemize 未装；英文 G2P 会退回 espeak-ng（需 apt install espeak-ng）"

echo "完成。激活：source $VENV/bin/activate"
echo "下一步：把 V1 模型放 models/zh-en/、SenseVoice 放 models/sense-voice/、vocab_tts.txt 放 models/（见 README）"
