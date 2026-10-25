# GPU 机会话环境——ssh 重连后临时变量全丢，这个文件 source 一次就够。
# 用法：. /root/env_gpu.sh   （bootstrap 会自动把它挂进 ~/.bashrc）
# 注意：本文件是被 source 的，不要 set -e；任何一步失败都不能中断登录 shell。

_ENV_DATA="${DATA:-${ENV_DATA:-/root}}"
[ -d "$_ENV_DATA/matcha-tts-finetune" ] || _ENV_DATA="/root"

REPO="$_ENV_DATA/matcha-tts-finetune"
ICEFALL="$_ENV_DATA/icefall"
COSY="$_ENV_DATA/CosyVoice"
TTS="$ICEFALL/egs/baker_zh/TTS"

# venv 不激活就会用 conda 的 python3 → k2/lhotse 全找不到（重连后第一大坑）
if [ -f "$REPO/.venv/bin/activate" ]; then
  . "$REPO/.venv/bin/activate"
  PY="$REPO/.venv/bin/python"
else
  PY="python3"
fi

# CosyVoice 不进来：它的源码目录里有 tools/，长期挂 PYTHONPATH 会遮蔽别的名。
# 跑老师合成时单独前缀 PYTHONPATH=$COSY 即可。本仓 scripts/ 也不进来（有 config.py
# 这种易撞名的顶层模块），直接 `python $REPO/scripts/xxx.py` 时它自己会入 path。
PYTHONPATH="$ICEFALL${PYTHONPATH:+:$PYTHONPATH}"
# python 的 logging 时间戳用北京时间，和控制台一致（容器本身是 UTC）
TZ="Asia/Shanghai"
# 输出经 tee/管道时 python 会缓冲，日志 tail 半天不动；这里强制行缓冲
PYTHONUNBUFFERED="1"
PIP_EXTRA_INDEX_URL="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export DATA REPO ICEFALL COSY TTS PY PYTHONPATH TZ PYTHONUNBUFFERED PIP_EXTRA_INDEX_URL

# 网页控制台的粘贴常触发 bracketed paste，把首行粘成 "[200~cat ..."
case $- in *i*) bind 'set enable-bracketed-paste off' 2>/dev/null ;; esac

if [ -n "${ENV_VERBOSE:-}" ]; then
  echo "env: PY=$($PY -V 2>&1)  REPO=$REPO  icefall=$([ -d "$ICEFALL" ] && echo ok || echo MISSING)"
fi
unset _ENV_DATA ENV_VERBOSE
