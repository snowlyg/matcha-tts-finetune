# matcha-tts-finetune

医院床旁/主机端侧 TTS（中英混读 Matcha `matcha-icefall-zh-en`）**微调工具链**：
PC 复现 + 内容错率尺子 + 语料生产 + icefall 训练/导出流程。
目标：把 `体温36.5度`、`为您自动接听` 这类**中文小数/轻声音节**的念错率降下来，
产物仍是**端侧可跑的 matcha ONNX**（词表/声码器不变，只换 `model-steps-3.onnx`）。

## 为什么是微调 Matcha（不是换模型）
端侧是 32 位 ARM、无 GPU、走 sherpa-onnx，只支持 VITS/Matcha/Kokoro。GPT-SoVITS/CosyVoice/
Pocket 等都跑不上端侧。详见 [docs/edge-deploy-constraints.md](docs/edge-deploy-constraints.md)。

## 目录
```
scripts/   g2p.py 复现前端+自检 | repro_v1.py PC合成 | ruler.py 错率尺子 | asr.py/config.py
           make_record_script.py 录制稿生成 | gen_teacher_corpus.py 合成语料+ASR门控
           score_speaker.py 发音筛选 | prepare_data.py 语料->Lhotse cuts
           make_train_tokens.py 训练词表 | make_warmstart_ckpt.py V1权重->epoch-1.pt
data/      record_script.tsv 录制稿 | record_script_ext.tsv 扩展稿
docs/      author-model-facts.md 模型取证 | icefall-finetune-flow.md 训练流程
           edge-deploy-constraints.md 端侧约束 | gpu-bootstrap.md 云实例搭建手册
cloud/     bootstrap_gpu.sh 云实例一键搭建 | run_finetune.sh 云端微调一键骨架
models/    (不入库) V1 ONNX + vocab_tts.txt + sense-voice —— 见 setup.sh
out/       (不入库) 合成 wav / 清单 / 日志
```

## 快速开始（PC，无需 GPU）
```
PYTHON=python3.10 bash setup.sh    # 建 venv + 装依赖（piper_phonemize 按平台自动选 wheel）
# 放模型到 models/：V1 的 model-steps-3.onnx + vocos-16khz-univ.onnx + vocab_tts.txt；
#                   SenseVoice 到 models/sense-voice/（地址见 docs/gpu-bootstrap.md）
python scripts/g2p.py --selftest   # 前端回归：unknown_diffs=0
python scripts/ruler.py --n 100    # 现装模型基线错率
```

## 语料与训练（要 GPU/云）
1. 云实例按 [docs/gpu-bootstrap.md](docs/gpu-bootstrap.md) 搭环境。
2. 语料二选一：`gen_teacher_corpus.py`（老师模型合成 + ASR 门控）或真人录制
   （`score_speaker.py` 先筛后录）。
3. `prepare_data.py` → 按 [docs/icefall-finetune-flow.md](docs/icefall-finetune-flow.md)
   或 `cloud/run_finetune.sh`：词表 → 前端 → 特征 → 热启动 V1 → 训练 → 导出 ONNX →
   `ruler.py` 验收。

## 红线
- baker 数据集非商用；作者 Pocket 权重 CC-BY-NC。医院是商用 → 只用自录/合成/Apache 数据，交付前过法务。
- 训练词表必须钉死作者那份，**不要跑 `generate_tokens.py`**；特殊 token 与热启动基座的取证结论
  见 [docs/author-model-facts.md](docs/author-model-facts.md)。
