# matcha-tts-finetune

医院床旁/主机端侧 TTS（中英混读 Matcha `matcha-icefall-zh-en`）**微调工具链**：
PC 复现 + 内容错率尺子 + 候选人发音筛选 + 语料准备 + icefall 训练/导出流程。
目标：把 `体温36.5度`、`为您自动接听` 这类**中文小数/轻声音节**的念错率降下来，
产物仍是**端侧可跑的 matcha ONNX**（词表/声码器不变，只换 `model-steps-3.onnx`）。

## 为什么是微调 Matcha（不是换模型）
端侧是 32 位 ARM、无 GPU、走 sherpa-onnx，只支持 VITS/Matcha/Kokoro。GPT-SoVITS/CosyVoice/
Pocket 等都跑不上端侧。详见 [docs/edge-deploy-constraints.md](docs/edge-deploy-constraints.md)。

## 目录
```
scripts/   g2p.py 复现前端+自检 | repro_v1.py PC合成 | ruler.py 错率尺子
           score_speaker.py 候选人筛选 | prepare_data.py 语料->Lhotse cuts | asr.py/config.py
data/      record_script.tsv 录制稿(60句)
docs/      author-model-facts.md | icefall-finetune-flow.md | edge-deploy-constraints.md
cloud/     run_finetune.sh 云端微调骨架
models/    (不入库) V1 ONNX + vocab_tts.txt + sense-voice —— 见 setup.sh
out/       (不入库) 合成 wav / 清单 / 日志
```

## 快速开始（PC，无需 GPU）
```
bash setup.sh                      # 建 venv + 装依赖（piper_phonemize 走 gh-proxy）
# 手动放模型到 models/：V1 的 model-steps-3.onnx + vocos-16khz-univ.onnx + tokens.txt
#                       vocab_tts.txt；SenseVoice 到 models/sense-voice/
python scripts/g2p.py --selftest   # 前端回归：unknown_diffs=0
python scripts/ruler.py --n 30     # 现装模型基线错率
```

## 录制与训练（要 GPU/云）
1. 候选人先念 20–30 句 → `scripts/score_speaker.py` 筛（错率 <10% 再录全量）。
2. 录 1–2h 自录语料（**不碰 baker**，商用授权）→ `scripts/prepare_data.py`。
3. 云端按 [docs/icefall-finetune-flow.md](docs/icefall-finetune-flow.md) 或 `cloud/run_finetune.sh`
   热启动 V1 pt → 训练 → 导出 ONNX → `ruler.py` 验收。

## 红线
- baker 数据集非商用；作者 Pocket 权重 CC-BY-NC。医院是商用 → 只用自录/Apache 数据，交付前过法务。
- 训练词表必须钉死作者那份，**不要跑 `generate_tokens.py`**；特殊 token(pad/blank) 待核，
  见 [docs/author-model-facts.md](docs/author-model-facts.md)。

## 基线（n=20，steps-3）
体温 30% / 为您自动接听 70% / 21床 0% / 字母读对。错在模型、可训。
