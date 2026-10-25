#!/usr/bin/env bash
# GPU 实例一键搭建（幂等，可反复跑、断点续跑）。
# 用法（AutoDL/恒源云等 Linux x86_64 实例，Ubuntu + conda/venv 均可）：
#   export DATA=/root/autodl-tmp        # 工作目录/数据盘挂载点，默认 /root
#   bash bootstrap_gpu.sh               # 或 curl 下来跑
# 任何一步失败：修完直接重跑，已完成的部分会跳过。
set -uo pipefail

DATA="${DATA:-/root}"
REPO_URL="https://github.com/snowlyg/matcha-tts-finetune.git"
MS_BASE="https://modelscope.cn/api/v1/models/dengcunqin/matcha_tts_zh_en_20251010/resolve/master"
SV_TGZ="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
GH="https://gh-proxy.com/"
PYBIN="${PYBIN:-python3}"
# pip 走国内镜像：download.pytorch.org 实测 ~350KB/s，TUNA 的 PyPI 镜像 ~38MB/s
PIP_MIRROR="${PIP_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# torch→torchvision 配套（钉住防 torchvision 要求 torch>=2.14 冲突）
tv_for() { case "$1" in
  2.7.*) echo 0.22.0;; 2.6.*) echo 0.21.0;; 2.5.*) echo 0.20.0;;
  2.4.*) echo 0.19.0;; 2.8.*) echo 0.23.0;; 2.9.*) echo 0.24.0;;
  *) echo "";; esac; }

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
# torch 全家桶 + CosyVoice 权重 ≈ 18G，盘紧先清 pip cache
AVAIL_G="$(df -BG --output=avail "$DATA" 2>/dev/null | tail -1 | tr -dc '0-9')"
[ -z "$AVAIL_G" ] || [ "$AVAIL_G" -ge 20 ] || \
  say "!! $DATA 仅剩 ${AVAIL_G}G（<20G 易装爆）：先 pip cache purge / 清旧 torch"
if "$PYBIN" -c 'import torch; exit(0 if torch.version.cuda and int(torch.version.cuda.split(".")[0]) >= 12 else 1)'; then :; else
  say "!! torch 的 CUDA 太老；Blackwell(50系) 需 cu128: pip install 'torch==2.7.0' --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128"; fi

cd "$DATA" || exit 1

# 1) 本仓 + venv
if [ ! -d matcha-tts-finetune ]; then
  say "== 1) clone 本仓 =="
  git clone "$REPO_URL" || git clone "${GH}${REPO_URL}"
fi
cd matcha-tts-finetune
# 工具脚本一律入库：重跑 bootstrap 时必须把仓更新到最新，否则实例上一直是旧脚本
# （以前这里靠"从 cloud/ 同目录携带 + 目标存在就跳过"，脚本入库后那条既多余又会压住新代码）。
say "== 1) git pull（已有仓）=="
git pull --ff-only || say "!! git pull 失败（本地有改动/网络不通），沿用当前树"
git rev-parse --short HEAD
HERE="$(cd "$(dirname "$0")" && pwd)"
# 参考素材不入库（音色 wav 属私有），只从 bootstrap 同目录或仓根取
for f in ref_v1.wav ref_v1.txt; do
  [ -f "$HERE/$f" ] && cp -u "$HERE/$f" .
done
MISSREF=""
for f in ref_v1.wav ref_v1.txt; do [ -f "$f" ] || MISSREF="$MISSREF $f"; done
[ -z "$MISSREF" ] || say "!! 缺参考素材:$MISSREF（在 $HERE 或仓根，用网页文件管理器上传后 cp 过来）"
if [ ! -d .venv ]; then
  say "== 1b) venv + 依赖（piper_phonemize 自动选 wheel）=="
  PYTHON="$PYBIN" bash setup.sh
fi
source .venv/bin/activate
pip config set global.index-url "$PIP_MIRROR" >/dev/null 2>&1 || true
# 镜像自带的 pip 配置常往 extra-index-url 塞 download.pytorch.org（~350KB/s）；用环境变量盖掉
export PIP_EXTRA_INDEX_URL="$PIP_MIRROR"

# 1c) 会话环境：ssh 重连后 export 的 PYTHONPATH / 激活的 venv / $TTS 全丢，
#     下一步就会以"conda 的 python3 + 没有 icefall"的姿态跑训练。装一个 source 文件。
if [ -f "$HERE/env_gpu.sh" ]; then
  cp -u "$HERE/env_gpu.sh" "$DATA/env_gpu.sh"
  grep -q "env_gpu.sh" ~/.bashrc 2>/dev/null || \
    echo ". $DATA/env_gpu.sh" >> ~/.bashrc
  say "已装 $DATA/env_gpu.sh（重连后自动生效；手动：. $DATA/env_gpu.sh）"
fi

# 2) 模型资产
say "== 2) V1 + SenseVoice 资产 =="
mkdir -p models/zh-en models/sense-voice
for f in model-steps-3.onnx vocos-16khz-univ.onnx vocab_tts.txt pytorch_model_sherpa.bin; do
  [ -s "models/zh-en/$f" ] || curl -L --max-time 900 -o "models/zh-en/$f" "$MS_BASE/$f"
done
[ -s models/vocab_tts.txt ] || cp models/zh-en/vocab_tts.txt models/vocab_tts.txt
if [ ! -s models/sense-voice/model.int8.onnx ]; then
  # gh-proxy 在部分实例近乎不可用（实测 ~8KB/s）；hf-mirror 是 CDN，通常快
  SV_URLS=(
    "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/${SV_TGZ}"
    "${GH}https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${SV_TGZ}")
  for u in "${SV_URLS[@]}"; do
    curl -L --max-time 900 -o "/tmp/$SV_TGZ" "$u" && tar xjf "/tmp/$SV_TGZ" -C models && break
  done
  [ -d "models/${SV_TGZ%.tar.bz2}" ] || { say "!! SenseVoice 下载失败，改用网页文件管理器上传 /tmp/$SV_TGZ"; exit 1; }
  mv models/${SV_TGZ%.tar.bz2}/{model.int8.onnx,tokens.txt} models/sense-voice/
  rm -rf "models/${SV_TGZ%.tar.bz2}" "/tmp/$SV_TGZ"
fi
python scripts/g2p.py --selftest || { say "!! 自检未过，先排查再往下"; exit 1; }

# 3) icefall + k2
say "== 3) icefall =="
cd "$DATA"
# icefall 用我们 fork 的 matcha-zh-en-16k 分支：微调必需的 5 处改动已经并进去
#   16k 特征提取 / 训练不插 blank / normalized_text 缺失时回退 .text /
#   导出元数据 sample_rate=16000 / 导出前用 --cmvn 覆写 model.mel_* 反归一化守卫。
# clone 它就不需要任何 sed；fork 不可用时退回上游再打 patches/ 里那份同名补丁。
IF_URL="${GH}https://github.com/snowlyg/icefall.git"
IF_BRANCH="matcha-zh-en-16k"
PATCH="$DATA/matcha-tts-finetune/patches/icefall-baker-zh-16k.patch"
if [ ! -d icefall ]; then
  git clone --depth 1 -b "$IF_BRANCH" "$IF_URL" \
    || { say "fork 拉取失败，退回上游 + 打补丁"; git clone --depth 1 https://github.com/k2-fsa/icefall.git \
         && (cd icefall && patch -p1 -N --forward < "$PATCH"); }
fi
grep -q "overriding with --cmvn" icefall/egs/baker_zh/TTS/matcha/export_onnx.py || \
  say "!! icefall 树里缺导出守卫（补丁第 5 处）——导出会沿用 ckpt 的 mel buffer，静默拉歪 mel 域"
grep -q "intersperse_blank=False" icefall/egs/baker_zh/TTS/matcha/train.py || \
  say "!! icefall 树里还在插 blank（补丁第 1 处没生效）"
source "$DATA/matcha-tts-finetune/.venv/bin/activate"
pip install -q -r icefall/requirements-tts.txt
# requirements-tts 不够：icefall.utils 还 import kaldialign / sentencepiece（在主 requirements.txt 里）。
# 一次装完，但滤掉 torch 家族 + k2 并用 constraints 钉住，否则 torch 会被拖走（同 CosyVoice 的教训）。
TORCH_NOW="$(python -c 'import torch;print(torch.__version__.split("+")[0])')"
printf 'torch==%s\ntorchaudio==%s\n' "$TORCH_NOW" "$TORCH_NOW" > /tmp/icefall_c.txt
grep -vE '^(torch|torchaudio|torchvision|triton|k2)([=<>!~-].*)?$' \
  icefall/requirements.txt > /tmp/icefall_req.txt
pip install -q -r /tmp/icefall_req.txt -c /tmp/icefall_c.txt
# baker_zh 配方的 local/ 脚本还要这两个（icefall 两份 requirements 都没覆盖到）：
#   jieba → convert_text_to_tokens.py 的分词；lilcom → compute_fbank 的 LilcomChunkyWriter
pip install -q jieba lilcom
# 配方的训练脚本靠 `import icefall`，但 clone 下来不等于可导入（也没 pip install -e）
# → 挂 PYTHONPATH 并写进 bashrc，否则 matcha/train.py 直接 ModuleNotFoundError
export PYTHONPATH="$DATA/icefall:$PYTHONPATH"
grep -q "icefall" ~/.bashrc 2>/dev/null || \
  echo "export PYTHONPATH=$DATA/icefall:\$PYTHONPATH" >> ~/.bashrc
# monotonic_align 是编译扩展（prepare.sh stage -1），没编译则 models.matcha_tts 导入即失败
if [ ! -e "$DATA/icefall/egs/baker_zh/TTS/matcha/monotonic_align/core.cpython-"*".so" ]; then
  say "编译 monotonic_align"
  ( cd "$DATA/icefall/egs/baker_zh/TTS/matcha/monotonic_align" && \
    python setup.py build_ext --inplace ) || { say "!! monotonic_align 编译失败"; exit 1; }
fi
python -c "import icefall, monotonic_align; print('icefall+monotonic_align ok')" || exit 1
# k2 不在 PyPI，且 import 时硬校验 torch 的补丁号（+cuda12.6.torch2.7.1 的轮子
# 拒绝跑在 2.7.0 上）→ 先把 torch 对齐到轮子版本，再 --no-deps 装 k2 保护 torch 家族。
# 副作用：torch 2.7.1 会拉 triton==3.3.0，与 openai-whisper 的 triton<3 冲突（仅警告，
# TTS 训练/推理不用 torch.compile）；千万别用 -c 去钉 triton，会 ResolutionImpossible。
K2_DIR="ubuntu-cuda/1.24.4.dev20260625"
K2_WHL="k2-1.24.4.dev20260625+cuda12.6.torch2.7.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
python -c "import k2" 2>/dev/null || {
  say "== 装 k2（训练必需）=="
  pip install -q "torch==2.7.1" "torchaudio==2.7.1" "torchvision==0.22.1"
  curl -fL --max-time 900 -o "/tmp/$K2_WHL" "https://huggingface.co/csukuangfj2/k2/resolve/main/$K2_DIR/$K2_WHL" \
    || curl -fL --max-time 900 -o "/tmp/$K2_WHL" "https://hf-mirror.com/csukuangfj2/k2/resolve/main/$K2_DIR/$K2_WHL" \
    || { say "!! k2 轮子下载失败：在 Mac 上下好后用网页文件管理器传到 /root"; exit 1; }
  pip install -q --no-deps "/tmp/$K2_WHL"   # 保留完整文件名，改名会 not a valid wheel filename
}
python -m k2.version || { say "!! k2 不可用，训练跑不了"; exit 1; }

# 4) CosyVoice + 权重
say "== 4) CosyVoice =="
cd "$DATA"
[ -d CosyVoice ] || git clone --recursive "${GH}https://github.com/FunAudioLLM/CosyVoice.git"
cd CosyVoice
# —— CosyVoice 依赖链是本轮最大的坑，逐条钉死，勿改顺序 ——
# ① requirements 里 torch 未钉版本 → 直接 -r 会把镜像自带的 torch 拉到最新 2.14.x
#    （连带 cu13 那套 nvidia 包），既慢又引爆 ②。先抓版本、写 constraints、过滤 torch 家族。
#    tensorrt 是部署加速项，推理合成用不到，且其 sdist 构建会 BackendUnavailable → 一并滤掉。
TORCH_V="$(python -c 'import torch;print(torch.__version__.split("+")[0])')"
TV_V="$(tv_for "$TORCH_V")"
# constraints 只钉 torch/torchaudio/torchvision —— **triton 千万别写进去**：
# torch 2.7.0 元数据硬依赖 triton==3.3.0，与 whisper 要的 <3 互斥，写进约束 = ResolutionImpossible。
# triton 由下面手动装 2.3.1，pip 只会警告不会阻断。
{ printf 'torch==%s\ntorchaudio==%s\n' "$TORCH_V" "$TORCH_V"
  [ -n "$TV_V" ] && printf 'torchvision==%s\n' "$TV_V"; } > /tmp/constraints.txt
grep -vE '^(torch|torchaudio|torchvision|triton|openai-whisper|tensorrt)([=<>!~-].*)?$' \
  requirements.txt > /tmp/cosyvoice_req.txt
# ② setuptools：>=81 删了 pkg_resources（openai-whisper 构建要），>=77 又是 torch 2.14 的硬要求
#    → 同 venv 里与老 whisper 无解。75.8.0 保留 pkg_resources 且满足 torch 2.7。
pip install -q --force-reinstall "setuptools==75.8.0" wheel
# ③ openai-whisper 20231117：--no-deps 阻止解析器回溯 torch（否则下十几 GB 轮子），
#    triton 降 2.3.1 满足 whisper<3（TTS 推理不用 torch.compile）。
pip install -q "triton==2.3.1" numba tiktoken more-itertools regex -c /tmp/constraints.txt
pip install -q --no-build-isolation --no-deps "openai-whisper==20231117"
# ④ 其余依赖：--no-deps 绕开解析器回溯（requirements 全是精确版本），-c 兜住 torch 家族。
#    注意 pyworld 无预编译轮子、需 g++ 编 sdist → 先确保系统有 C++ 编译器。
#    精简镜像默认 apt 源直连 archive.ubuntu.com（极慢），先换国内镜像再装。
have g++ || {
  say "安装 g++（pyworld 编译需要）"
  sed -i 's|https\?://[a-z0-9.-]*\(archive\|security\)\.ubuntu\.com|https://mirrors.aliyun.com|g' \
    /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null
  apt-get update -qq && apt-get install -y -qq g++
}
pip install -q --no-deps -r /tmp/cosyvoice_req.txt -c /tmp/constraints.txt
# pydantic 是 pydantic-core 的纯 Python 壳，--no-deps 会造成两者版本错位
# （validate_core_schema ImportError）→ 带依赖重装一次，把 requirements 钉的 core 拉回来。
PYD_PIN="$(grep -iE '^pydantic=' requirements.txt | head -1)"
[ -n "$PYD_PIN" ] && pip install -q --force-reinstall "$PYD_PIN"
pip install -q ninja   # deepspeed 需要
# --no-deps 的补账循环：import 报错即清单，缺什么装什么（改名包查小表，其余按原名装）
imp2pkg() { case "$1" in
  ruamel) echo "ruamel.yaml<0.19";; cv2) echo opencv-python;; PIL) echo pillow;;
  sklearn) echo scikit-learn;; yaml) echo pyyaml;; fitz) echo pymupdf;;
  cpuinfo) echo py-cpuinfo;; pynvml) echo nvidia-ml-py;; *) echo "$1";; esac; }
python -c "import torch,whisper;print('torch',torch.__version__,'whisper',whisper.__version__)"
for i in 1 2 3 4 5 6; do
  ERR="$(PYTHONPATH=$PWD python -c "from cosyvoice.cli.cosyvoice import CosyVoice2" 2>&1 >/dev/null)" && break
  MISS="$(printf '%s' "$ERR" | grep -oE "No module named '[^']+'" | tail -1 | sed "s/.*'\([^']*\)'.*/\1/" | cut -d. -f1)"
  [ -n "$MISS" ] || { printf '%s\n' "$ERR"; say "!! cosyvoice 导入失败（非缺包），贴报错排查"; exit 1; }
  say "补装传递依赖: $(imp2pkg "$MISS")（第 $i 轮）"
  pip install -q -c /tmp/constraints.txt "$(imp2pkg "$MISS")" || { say "!! $(imp2pkg "$MISS") 安装失败"; exit 1; }
done
# 循环耗尽仍未 break → 最后实判一次，避免误报 ok
PYTHONPATH=$PWD python -c "from cosyvoice.cli.cosyvoice import CosyVoice2" \
  || { say "!! 补账 6 轮后 cosyvoice 仍导入失败，贴报错排查"; exit 1; }
say "cosyvoice import ok ✓"
# 权重：ModelScope 上的 ID 是 iic/CosyVoice2-0.5B（FunAudioLLM 是 HF 命名）
[ -d pretrained_models/iic/CosyVoice2-0.5B ] || {
  pip install -q modelscope
  modelscope download --model iic/CosyVoice2-0.5B --local_dir pretrained_models/iic/CosyVoice2-0.5B
}
# 子模块空壳检测（镜像自带 CosyVoice 时 third_party/Matcha-TTS 可能为空目录）
if [ ! -f third_party/Matcha-TTS/setup.py ] && [ ! -f third_party/Matcha-TTS/pyproject.toml ]; then
  say "!! third_party/Matcha-TTS 缺内容：unzip Matcha-TTS.zip 后 mv 到该目录，重跑本脚本"
  exit 1
fi
pip install -q -e third_party/Matcha-TTS || {
  pip install -q cython && pip install -q -e third_party/Matcha-TTS; }

# 5) 老师对照实验（60 条；ref 素材需先从本地 scp ref_v1.wav/ref_v1.txt 过来）
say "== 5) 就绪。先验这台机器：bash \$REPO/cloud/smoke.sh（全链冒烟，不需要真语料）=="
say "==    要产语料则参考素材未就位先 scp，然后： =="
cat <<EOF
cd $DATA/matcha-tts-finetune && source .venv/bin/activate
PYTHONPATH=$DATA/CosyVoice python scripts/gen_teacher_corpus.py --backend cosyvoice \\
  --model-dir $DATA/CosyVoice/pretrained_models/iic/CosyVoice2-0.5B \\
  --ref-wav ref_v1.wav --ref-text "\$(cat ref_v1.txt)" \\
  --script data/record_script_ext.tsv --limit 60 --n 1 --out out/teacher_cmp
EOF
