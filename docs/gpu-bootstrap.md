# 云 GPU 实例选型与开机手册（2026-10 实测行情）

任务画像：CosyVoice2-0.5B 合成（~8G 显存）+ matcha 微调 74M（16–24G 舒服）+
SenseVoice 门控（CPU 即可）。**单卡、断续、总 10–15 GPU 时/轮，预算 ¥100–300。**

## 1. 平台与计费
- 计费选**按量**（关机停计费），不要包日/包周（任务断续，闲置照扣）。
- 国内优先（支付宝充值、ModelScope 直连、下载快）：
  AutoDL / 恒源云 / 算家云 / 智星云 / 晨涧云 / 矩池云 / 优云智算(compshare) / UCloud。
  2026-10 实测：消费级卡普遍缺货，有货 > 便宜；¥2.5/时 内都算合理价。
- 备胎：RunPod（Community 4090 $0.34–0.69/h，Secure 与 Community 库存独立）、
  Vast.ai（$0.3–0.5 可中断，适合可重跑的合成任务）。需 Visa/万事达；
  权重改走 HuggingFace/hf-mirror。
- 错峰：凌晨 0–6 点、工作日上午 9–11 点常放库存。

## 2. 镜像规则（重要，选错白折腾）
| 项 | 要求 | 原因 |
|---|---|---|
| Python | **3.10**（3.11 凑合） | CosyVoice 官方 3.10；piper_phonemize cp310 wheel 齐全 |
| torch | **≥2.2** | 2.1 的 SDPA 导出 ONNX 报错（flow 文档第 6 步） |
| CUDA | ≥12.1；**5090/Blackwell 必须 ≥12.8 + torch≥2.7** | 旧构建无 sm_120 内核，报 no kernel image |
| 盘 | 环境/权重/checkpoint 全放数据盘 | 恒源云系统盘仅 20G |
| 保留期 | 恒源云关机仅保留 **72h**（AutoDL 15 天） | 当天产物当天 scp 回本地 |
| pip 源 | index 设国内镜像（TUNA/Aliyun），别让 extra-index 指向 download.pytorch.org | 实测 TUNA ~38MB/s、官方 pytorch 源 ~350KB/s |
| k2 | 镜像自带 k2 最省事；否则要按轮子的 torch 补丁号对齐 | 训练必需，且 k2 不在 PyPI |

## 3. 开机后验证（30 秒）
```bash
nvidia-smi | head -12
python -V
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
df -h | grep -vE 'tmpfs|overlay$'    # 确认数据盘挂载点，下文记 $DATA
```
torch 与卡不匹配（no kernel image）→ 走国内镜像重装配套版本，
`pip install "torch==2.7.0" "torchaudio==2.7.0" -i https://pypi.tuna.tsinghua.edu.cn/simple`
（PyPI 默认 torch 2.7.0 即 CUDA 构建，实测 `2.7.0+cu126`；cu128 专用构建走
`--index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128`。
**别用 download.pytorch.org**，国内实例 ~350KB/s，一个 torch 轮子要下几十分钟。）

**每次重连/重启后的第一件事**：`. $DATA/env_gpu.sh`（bootstrap §1c 已把它挂进 `~/.bashrc`）。
`source .venv/bin/activate`、`export PYTHONPATH`、`$TTS` 都是会话级变量，新 ssh 会话里全丢，
此时 `python` 落回镜像自带的 conda 解释器，训练脚本会报
`ModuleNotFoundError: No module named 'k2'` —— 看着像依赖没装好，其实只是没激活 venv。
判断方法：`which python` 必须是 `$DATA/matcha-tts-finetune/.venv/bin/python`。

## 4. 环境搭建
一键版（幂等、可断点续跑）。**工具脚本已全部入库**，实例靠 `git pull` 就能拿到全套；
需要 scp 的只有"拿到仓之前就得存在"的两个引导文件和不入库的参考音色：
```bash
# 在 Mac 上（实例路径按平台调整，AutoDL 用 root@区域端口）：
ssh -p <端口> root@<区域> 'mkdir -p /root/tools'
scp -p -P <端口> cloud/bootstrap_gpu.sh cloud/env_gpu.sh \
  out/ref_v1.wav out/ref_v1.txt root@<区域>:/root/tools/
# 在实例上：
export DATA=/root/autodl-tmp     # 按 df -h 确认的实际挂载点改；默认 /root（无数据盘的实例直接用）
bash /root/tools/bootstrap_gpu.sh    # 内含 clone/pull 本仓 + 拉 icefall fork 分支
# 建完立刻验这台机器（不需要真语料，3~6 分钟）：
bash $DATA/matcha-tts-finetune/cloud/smoke.sh
```
之后日常只改本地、`git push`，实例上 `cd $REPO && git pull` 即可；
`cloud/session_fix_mel.sh`、`cloud/fix_mel_buffer.py`、`scripts/*` 都跟着仓走，不再 scp。
手动版（逐步核对用）：
```bash
cd $DATA && git clone https://github.com/snowlyg/matcha-tts-finetune.git
cd matcha-tts-finetune && bash setup.sh && source .venv/bin/activate

# 模型资产（V1 从 ModelScope；SenseVoice 从 gh-proxy）
mkdir -p models/zh-en models/sense-voice
BASE="https://modelscope.cn/api/v1/models/dengcunqin/matcha_tts_zh_en_20251010/resolve/master"
for f in model-steps-3.onnx vocos-16khz-univ.onnx vocab_tts.txt pytorch_model_sherpa.bin; do
  curl -L --max-time 600 -o models/zh-en/$f $BASE/$f; done
cp models/zh-en/vocab_tts.txt models/vocab_tts.txt
cd models && curl -L -O https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2 \
  && tar xjf *.tar.bz2 && mv sherpa-onnx-sense-voice-*/{model.int8.onnx,tokens.txt} sense-voice/ \
  && rm -rf sherpa-onnx-sense-voice-* *.tar.bz2 && cd ..
python scripts/g2p.py --selftest    # 门槛：unknown_diffs=0

# icefall + CosyVoice
cd $DATA
git clone https://gh-proxy.com/https://github.com/k2-fsa/icefall.git
pip install -r icefall/requirements-tts.txt
# k2（训练必需，不在 PyPI）：先按轮子的 torch 补丁号对齐，再 --no-deps 装
pip install "torch==2.7.1" "torchaudio==2.7.1" "torchvision==0.22.1"
curl -fL -O "https://huggingface.co/csukuangfj2/k2/resolve/main/ubuntu-cuda/1.24.4.dev20260625/k2-1.24.4.dev20260625+cuda12.6.torch2.7.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
pip install --no-deps ./k2-*.whl && python -m k2.version
git clone --recursive https://gh-proxy.com/https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
# 依赖链（勿改顺序，原因见下方"已知坑"）：constraints 只钉 torch 家族、不含 triton
printf 'torch==2.7.0\ntorchaudio==2.7.0\ntorchvision==0.22.0\n' > /tmp/constraints.txt
grep -vE '^(torch|torchaudio|torchvision|triton|openai-whisper|tensorrt)([=<>!~-].*)?$' \
  requirements.txt > /tmp/req2.txt
pip install --force-reinstall "setuptools==75.8.0" wheel
pip install "triton==2.3.1" numba tiktoken more-itertools regex -c /tmp/constraints.txt
pip install --no-build-isolation --no-deps "openai-whisper==20231117"
apt-get install -y g++   # pyworld 无预编译轮子，sdist 需要 g++；慢则先把 apt 源换成 mirrors.aliyun.com
pip install --no-deps -r /tmp/req2.txt -c /tmp/constraints.txt
pip install "ruamel.yaml==0.18.17" "huggingface-hub<1.0" "fsspec<2025" "tokenizers<0.22,>=0.21"  # --no-deps 的补账
pip install --force-reinstall "pydantic==2.7.0" ninja  # pydantic/pydantic-core 版本错位修复
PYTHONPATH=$DATA/CosyVoice python -c "from cosyvoice.cli.cosyvoice import CosyVoice2; print('import ok')"
pip install modelscope
modelscope download --model iic/CosyVoice2-0.5B --local_dir pretrained_models/iic/CosyVoice2-0.5B
```
已知坑：
- **icefall clone ≠ 可导入，且 requirements-tts 不全**：配方的 `matcha/train.py` 里
  `from icefall.utils import ...` → 要 `export PYTHONPATH=$DATA/icefall:$PYTHONPATH`；
  而 `icefall.utils` 还 `import kaldialign`、`import sentencepiece`，这两个只在主
  `requirements.txt` 里 → 一次装完但**滤掉 torch 家族和 k2 并配 constraints**
  （`grep -vE '^(torch|torchaudio|torchvision|triton|k2)...' requirements.txt`），否则 torch 又被拖走。
  同理 `matcha/monotonic_align` 是编译扩展（`prepare.sh` stage -1），不先
  `python setup.py build_ext --inplace` 会在 `import models.matcha_tts` 处报
  `No module named 'monotonic_align.core'`。
- **k2 不在 PyPI**：训练必需（icefall matcha 用 `k2.SymbolTable`）。轮子在 HF `csukuangfj2/k2`，
  且 import 时**按 torch 补丁号硬校验**——`+cuda12.6.torch2.7.1` 的轮子跑在 torch 2.7.0 上直接
  `ImportError: k2 was built using PyTorch 2.7.1`。→ 先把 torch/torchaudio/torchvision 对齐到轮子
  的版本，再 `pip install --no-deps <完整文件名>.whl`。副作用：升 torch 会把 triton 拉回 3.3.0，
  openai-whisper 的 `triton<3` 警告可忽略（别用 `-c` 钉 triton，会 `ResolutionImpossible`）。
- `monotonic_align` 编译错 → 先 `pip install cython` 再 `pip install -e third_party/Matcha-TTS`；
  piper_phonemize wheel 必须保留完整文件名再 pip 装（否则 not a valid wheel filename）。
- **CosyVoice requirements 的 torch 未钉版本**：直接 `pip install -r requirements.txt` 会把镜像
  自带的 torch 拉到最新 2.14.x（连带 cu13 那套 nvidia 包，数 GB）。→ 过滤 torch 家族 + constraints 钉死。
- **setuptools 死结**：setuptools≥81 删了 `pkg_resources`，openai-whisper 20231117 构建要它；
  而 torch 2.14 又要求 setuptools≥77 → 同 venv 无解。→ torch 钉 2.7 + setuptools 钉 **75.8.0**。
- **whisper↔triton 回溯**：whisper 20231117 要 `triton<3`，torch 2.7.0 要 `triton==3.3.0`，
  直接装 whisper 会让 pip 从 2.14 一路回溯下载十几 GB torch 轮子。→ `--no-deps` 装 whisper、
  triton 单独降 2.3.1（TTS 推理不用 torch.compile，无影响）。
- **constraints 里不要写 triton**：torch 2.7.0 元数据硬依赖 `triton==3.3.0`，约束同时含
  `torch==2.7.0` + `triton==2.3.1` 时 pip 判 `ResolutionImpossible` 并继续回溯。→ 约束只钉
  torch/torchaudio/torchvision；req2 用 `--no-deps` 绕开解析器（requirements 全是精确版本），
  传递依赖缺失由 import 检查兜底、按 ModuleNotFoundError 单独补装
  （已踩：hyperpyyaml→ruamel<0.19、镜像预装 huggingface-hub 1.x↮transformers 4.51、lightning→fsspec<2025、
  transformers→tokenizers、deepspeed→py-cpuinfo/nvidia-ml-py/hjson/ninja——注意 import 名≠包名：cpuinfo→py-cpuinfo；
  pydantic 与 pydantic-core 会版本错位（validate_core_schema ImportError）→ 带依赖 --force-reinstall pydantic）。
- **cosyvoice 不是 pip 包**：它是 `$DATA/CosyVoice` 源码目录，从别处（如训练仓目录）跑脚本会
  `No module named 'cosyvoice'` → 命令前加 `PYTHONPATH=$DATA/CosyVoice`。
- **tensorrt-cu12**：部署加速项，推理合成用不到，其 sdist 构建报 `BackendUnavailable` → 从 req 滤掉。
- **pyworld 编译**：无 cp310 预编译轮子，sdist 要 C++ 编译器，精简镜像里常缺 →
  `apt-get install -y g++`；且一批 `-r` 安装里任何一个构建失败会**整批回滚**（表现为后续
  ModuleNotFoundError 像是没装齐，其实是回滚）。
- **pip 多源**：镜像常把 download.pytorch.org 塞进 extra-index-url（~350KB/s）→
  `pip config set global.index-url <国内镜像>` 并 `export PIP_EXTRA_INDEX_URL=<国内镜像>` 盖掉。
- **heredoc 粘贴**：终端开着 bracketed-paste 时，粘贴 `cat <<EOF` 首行会变成 `[200~cat: command not found`
  → 写文件用 `printf '...\n' > 文件` 代替。
- **盘**：50G 系统盘装两套 torch（2.14+2.7）会爆 → `pip cache purge`，旧 torch 用 `pip uninstall` 别 `rm`。
- 镜像自带 CosyVoice 时 `third_party/Matcha-TTS` 可能是空壳 → 查 setup.py，缺则上传 Matcha-TTS.zip 解压。

## 5. 老师对照实验（当天要出的结论）
参考音色素材在本地仓已备好：`out/ref_v1.wav`（7.6s，三条 ASR 验证正确的 V1 体温句）
+ `out/ref_v1.txt`，scp 上云后：
```bash
cd $DATA/matcha-tts-finetune && source .venv/bin/activate
PYTHONPATH=$DATA/CosyVoice python scripts/gen_teacher_corpus.py --backend cosyvoice \
  --model-dir $DATA/CosyVoice/pretrained_models/iic/CosyVoice2-0.5B \
  --ref-wav ref_v1.wav --ref-text "$(cat ref_v1.txt)" \
  --script data/record_script_ext.tsv --limit 60 --n 1 --out out/teacher_cmp
```
判定：`report.log` 错率 **<5% 才放大产语料**（V1 基线 ~28%）；抽 5 条人耳听音色。
不达标 → 换 GPT-SoVITS 后端再测（脚本加 backend 即可）。

## 6. 收工纪律
```bash
scp -r <实例>:$DATA/matcha-tts-finetune/out/... ./out/    # report/corpus/checkpoint 当天回传
```
关机 ≠ 备份：恒源云 72 小时回收。训练阶段 `epoch-*.pt` 只认本地+网盘两份都算数。

## 7. 换厂商重建清单（按 2026-10 这轮实际踩过的点）
换云实例时**差异不在"装什么包"，在下面这些静默失效点**。顺序照抄，别跳。

1. **先体检再动手**（`cloud/bootstrap_gpu.sh` §0 已自动化）：GPU 算力 vs torch 的 CUDA 构建
   （Blackwell 需 cu128）、python ≥3.10、`df` 可用 ≥20G、`git`/`g++` 是否在、
   **默认 `python3` 落在哪个解释器**（镜像常把 conda base 放 PATH 最前，脚本会 import 不到 venv 的包）。
2. **会话环境**：`. $DATA/env_gpu.sh`（已挂进 `~/.bashrc`）；每次重连后先 `which python` 复验。
   这一条本轮换过一次跟头：掉线后 `python` 回到 conda，报 `No module named 'k2'`，看着像依赖没装。
3. **网络实测再选路**：`gh-proxy` 在不同厂商可从 ~8KB/s 变成不可用；HF/ModelScope/TUNA 谁通要
   用 `curl -w '%{speed_download}'` 实测一次再决定。pip 一律 `config set global.index-url` +
   `export PIP_EXTRA_INDEX_URL=<镜像>`（镜像常往 extra-index 塞 download.pytorch.org，~350KB/s）。
4. **依赖链固定顺序**（勿改，理由见"已知坑"）：torch 家族先钉 → constraints 只钉 torch 家族
   （**triton 绝不进 constraints**）→ k2 轮子（按 torch 补丁号硬校验、`--no-deps` 装）→
   icefall requirements 过滤 torch 家族 → CosyVoice 教师链。
5. **icefall 走 fork 分支** `snowlyg/icefall` 的 `matcha-zh-en-16k`（5 处必需改动已在树里）；
   厂商拉不动 GitHub 时用 `patch -p1 -N --forward < patches/icefall-baker-zh-16k.patch`，
   或把两份仓镜像到 gitcode 后从国内站 clone。**绝不在已有数据的 icefall 树里重新 clone**——
   `data/`（cuts、特征、cmvn、tokens）和 `matcha/exp-*/` 都在树内，重拉等于删产物。
6. **数据路径**：`prepare_data.py` 写进 cuts 的是 wav **绝对路径** → 语料必须先落到本机、
   再在训练机上生成 cuts；换机器 cuts 即失效。
7. **关机保留性要实测、不要假设**：本轮 compshare 的 `/root` 关机过夜后仍在（实测），
   但别的厂商可能是临时盘。新实例第一天先做一次"关机 10 分钟再开"验证，再决定要不要每天回传。
8. **时间**：容器常是 UTC、控制台常显示北京时间，两者不可互相推断；进度与 ETA 一律用
   **日志时间戳 + 实测速率**算（本轮实测 4.7 秒/epoch），别拿实例 `date` 去推。
9. **计费纪律**：长任务 `nohup` + 手动关机；控制台定时关机不可信（本轮设了 02:36 没触发，多烧 ~7h）。
    ssh 掉线不杀任务靠 nohup，不靠 tmux 重连。
10. **新厂商第一天必须走一遍全链冒烟**：`bash cloud/smoke.sh`（约 3~6 分钟，不需要真语料——
    它自己合成 16 句占位波形把整条管线跑完）。检查项依次为 venv/import、五处 icefall 改动、
    资产、数据链(cuts→tokens→fbank→cmvn)、热启动、A/B loss 对照、导出物结构校验
    （`sample_rate=16000`、无 `add_blank`、**mel 常数必须等于训练那对**）、出声门控、元数据覆盖。
    这轮最贵的教训是门控放错位置——loss 一路降而 ruler 92.2%，所以冒烟的终点是
    "导出物里 mel 那对与 `--cmvn` 一致 + 这台机器能合成能回读"，**不是 loss**。
    注意冒烟 [8] 的出声门控用的是 V1 本身，验的是工具链；它给出的错率就是**这台机器的基线**，
    正式验收必须与它同机比（跨机器 ASR/线程数不可比）。
11. **补丁能不能打上，用 dry-run 实测，不要假设**：
    `cd $ICEFALL && patch -p1 --dry-run -N --forward < $REPO/patches/icefall-baker-zh-16k.patch`
    （`-p1` 的基准是 icefall 仓根，在 `egs/baker_zh/TTS` 里跑会报 "File to patch"）。

## 8. 产物回传约定（云端 → 本地）
| 要什么 | 在哪 | 为什么要 |
|---|---|---|
| 候选 ONNX | `cand/steps-3-*.onnx`（打包 `tar czf`） | 本地 ruler 对基线判错率 |
| 训练日志 | `/tmp/ft*.log`、`/tmp/export-*.log` | 曲线、warning（尤其"overriding with --cmvn"） |
| ckpt | 只取要用的那几个（~214MB/个） | 复现/继续微调 |
| 数据侧 | `data/fbank/cmvn.json`、`data/tokens.txt`、`git rev-parse --short HEAD` | 域参数与代码版本可追溯 |
| 语料 | `out/corpus1k.tgz` | 本地留唯一副本 |

文件名要带 epoch 与厂商标识（如 `steps-3-melfix-ep300-compshare.onnx`）——同名覆盖本踩过一次：
icefall 导出物写死 `model-steps-2..6.onnx` 落在 CWD，连导三个 epoch 后只剩最后一个。
