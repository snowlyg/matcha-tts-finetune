# matcha-tts-finetune

中英混读 Matcha TTS 模型（`matcha-icefall-zh-en`，icefall baker_zh Matcha + vocos-16k）的
**微调工具链**：前端复现与自检、内容错率尺子、语料生产（合成门控/真人筛选）、
icefall 热启动训练与 ONNX 导出、错率验收。
微调产物与基座的词表/声码器/token id 完全兼容，只替换声学模型 ONNX。

## 目录
```
scripts/   g2p.py 复现前端+自检 | repro_v1.py PC合成 | ruler.py 错率尺子 | asr.py/config.py
           make_record_script.py 录制稿生成 | gen_teacher_corpus.py 合成语料+ASR门控
           score_speaker.py 发音筛选 | score_wavs.py 给设备产出的 wav 打分
           prepare_data.py 语料->Lhotse cuts
           make_train_tokens.py 训练词表 | make_warmstart_ckpt.py V1权重->epoch-1.pt
  device/    端侧取证台架（标点矩阵 / 同句 N 次 A/B / 客观停顿）→ 见 scripts/device/README.md
  textlayer/ 书面边界改写的回归台（跑调用方仓库的真 .ets）→ 见 scripts/textlayer/README.md
  zipvoice/  ZipVoice 端侧判死的尺子（C 驱动 + OHOS 交叉编译 + 同机 RTF 对照）
data/      record_script.tsv 录制稿 | record_script_ext.tsv 扩展稿
           punct_cases.tsv 标点/边界实验话术（含"期望读音"列）| edge_baseline.txt 同机分母 20 句
docs/      author-model-facts.md 基座取证 | icefall-finetune-flow.md 训练流程
           device-runtime-check.md 端侧真机取证 | edge-deploy-constraints.md 端侧推理约束
           gpu-bootstrap.md 云实例搭建手册
cloud/     run_finetune.sh 云端微调一键骨架
models/    (不入库) V1 ONNX + vocab_tts.txt + sense-voice —— 见 setup.sh
out/       (不入库) 合成 wav / 清单 / 日志
```

## 快速开始
```
PYTHON=python3.10 bash setup.sh    # 建 venv + 装依赖（piper_phonemize 按平台自动选 wheel）
# 放模型到 models/：model-steps-3.onnx + vocos-16khz-univ.onnx + vocab_tts.txt；
#                   SenseVoice 到 models/sense-voice/（地址见 docs/gpu-bootstrap.md）
python scripts/g2p.py --selftest   # 前端回归：unknown_diffs=0
python scripts/ruler.py --n 100    # 基座内容错率
```

## 语料与训练
1. 云实例环境见 [docs/gpu-bootstrap.md](docs/gpu-bootstrap.md)。
2. 语料二选一：`gen_teacher_corpus.py`（老师模型合成 + ASR 门控）或真人录制
   （`score_speaker.py` 先筛后录）。
3. `prepare_data.py` → 按 [docs/icefall-finetune-flow.md](docs/icefall-finetune-flow.md)
   或 `cloud/run_finetune.sh`：词表 → 前端 → 特征 → 热启动 → 训练 → 导出 ONNX →
   `ruler.py` 验收。

## 相关文档
- [docs/author-model-facts.md](docs/author-model-facts.md)：基座取证（词表/特殊 token/权重/特征域）
- [docs/edge-deploy-constraints.md](docs/edge-deploy-constraints.md)：端侧推理约束与产物兼容
- [docs/device-runtime-check.md](docs/device-runtime-check.md)：端侧真机取证（标点存活矩阵、拆段的真实代价）
- [docs/icefall-finetune-flow.md](docs/icefall-finetune-flow.md)：训练与导出流程
- [docs/gpu-bootstrap.md](docs/gpu-bootstrap.md)：云实例选型与环境搭建
