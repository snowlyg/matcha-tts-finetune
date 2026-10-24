#!/usr/bin/env bash
# GPU 实例一键搭建（幂等，可反复跑、断点续跑）。
# 用法（AutoDL/恒源云等 Linux x86_64 实例，Ubuntu + conda/venv 均可）：
#   export DATA=/root/autodl-tmp        # 数据盘挂载点，默认 AutoDL 路径
#   bash bootstrap_gpu.sh               # 或 curl 下来跑
# 任何一步失败：修完直接重跑，已完成的部分会跳过。
set -uo pipefail

DATA="${DATA:-/root/autodl-tmp}"
REPO_URL="https://github.com/snowlyg/matcha-tts-finetune.git"
MS_BASE="https://modelscope.cn/api/v1/models/dengcunqin/matcha_tts_zh_en_20251010/resolve/master"
SV_TGZ="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
GH="https://gh-proxy.com/"
PYBIN="${PYBIN:-python3}"

say() { echo "[bootstrap] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# 0) 环境体检
say "== 0) 体检 =="
"$PYBIN" - <<'EOF' || true
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "capability", torch.cuda.get_device_capability() if torch.cuda.is_available() else "NO-GPU")
EOF
"$PYBIN" -c 'import sys; assert sys.version_info[:2] >= (3, 10), "需要 py>=3.10（CosyVoice 推荐 3.10/3.11）"' || exit 1
df -h "$DATA" || { say "数据盘 $DATA 不存在，先 export DATA=<挂载点>"; exit 1; }
if "$PYBIN" -c 'import torch; exit(0 if torch.version.cuda and int(torch.version.cuda.split(".")[0]) >= 12 else 1)'; then :; else
  say "!! torch 的 CUDA 太老；Blackwell(50系) 需 cu128: pip install torch --index-url https://download.pytorch.org/whl/cu128"; fi

cd "$DATA" || exit 1

# 1) 本仓 + venv
if [ ! -d matcha-tts-finetune ]; then
  say "== 1) clone 本仓 =="
  git clone "$REPO_URL" || git clone "${GH}${REPO_URL}"
fi
cd matcha-tts-finetune
if [ ! -d .venv ]; then
  say "== 1b) venv + 依赖（piper_phonemize 自动选 wheel）=="
  PYTHON="$PYBIN" bash setup.sh
fi
source .venv/bin/activate

# 2) 模型资产
say "== 2) V1 + SenseVoice 资产 =="
mkdir -p models/zh-en models/sense-voice
for f in model-steps-3.onnx vocos-16khz-univ.onnx vocab_tts.txt pytorch_model_sherpa.bin; do
  [ -s "models/zh-en/$f" ] || curl -L --max-time 900 -o "models/zh-en/$f" "$MS_BASE/$f"
done
[ -s models/vocab_tts.txt ] || cp models/zh-en/vocab_tts.txt models/vocab_tts.txt
if [ ! -s models/sense-voice/model.int8.onnx ]; then
  curl -L --max-time 900 -o "/tmp/$SV_TGZ" "${GH}https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$SV_TGZ"
  tar xjf "/tmp/$SV_TGZ" -C models
  mv models/${SV_TGZ%.tar.bz2}/{model.int8.onnx,tokens.txt} models/sense-voice/
  rm -rf "models/${SV_TGZ%.tar.bz2}" "/tmp/$SV_TGZ"
fi
python scripts/g2p.py --selftest || { say "!! 自检未过，先排查再往下"; exit 1; }

# 3) icefall
say "== 3) icefall =="
cd "$DATA"
[ -d icefall ] || git clone "${GH}https://github.com/k2-fsa/icefall.git"
source "$DATA/matcha-tts-finetune/.venv/bin/activate"
pip install -q -r icefall/requirements-tts.txt

# 4) CosyVoice + 权重
say "== 4) CosyVoice =="
cd "$DATA"
[ -d CosyVoice ] || git clone --recursive "${GH}https://github.com/FunAudioLLM/CosyVoice.git"
cd CosyVoice
pip install -q -r requirements.txt
[ -d pretrained_models/FunAudioLLM/CosyVoice2-0.5B ] || {
  pip install -q modelscope
  modelscope download --model FunAudioLLM/CosyVoice2-0.5B --local_dir pretrained_models/FunAudioLLM/CosyVoice2-0.5B
}
pip install -q -e third_party/Matcha-TTS || {
  pip install -q cython && pip install -q -e third_party/Matcha-TTS; }

# 5) 老师对照实验（60 条；ref 素材需先从本地 scp ref_v1.wav/ref_v1.txt 过来）
say "== 5) 就绪。参考素材未就位则先 scp，然后： =="
cat <<EOF
cd $DATA/matcha-tts-finetune && source .venv/bin/activate
python scripts/gen_teacher_corpus.py --backend cosyvoice \\
  --model-dir $DATA/CosyVoice/pretrained_models/FunAudioLLM/CosyVoice2-0.5B \\
  --ref-wav ref_v1.wav --ref-text "\$(cat ref_v1.txt)" \\
  --script data/record_script_ext.tsv --limit 60 --n 1 --out out/teacher_cmp
EOF
