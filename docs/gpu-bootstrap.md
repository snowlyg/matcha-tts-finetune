# 云 GPU 实例选型与开机手册（2026-10 实测行情）

任务画像：CosyVoice2-0.5B 合成（~8G 显存）+ matcha 微调 74M（16–24G 舒服）+
SenseVoice 门控（CPU 即可）。**单卡、断续、总 10–15 GPU 时/轮，预算 ¥100–300。**

## 1. 平台与计费
- 计费选**按量**（关机停计费），不要包日/包周（任务断续，闲置照扣）。
- 国内优先（支付宝、ModelScope 直连、数据不出境——医院商用项目合规更稳）：
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

## 3. 开机后验证（30 秒）
```bash
nvidia-smi | head -12
python -V
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
df -h | grep -vE 'tmpfs|overlay$'    # 确认数据盘挂载点，下文记 $DATA
```
torch 与卡不匹配（no kernel image）→
`pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128`

## 4. 环境搭建
一键版（幂等、可断点续跑，任何一步失败修完直接重跑）：
```bash
export DATA=/root/autodl-tmp     # 按 df -h 确认的实际挂载点改
cd $DATA && curl -L -O https://gh-proxy.com/https://raw.githubusercontent.com/snowlyg/matcha-tts-finetune/main/cloud/bootstrap_gpu.sh
bash bootstrap_gpu.sh
```
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
git clone --recursive https://gh-proxy.com/https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice && pip install -r requirements.txt && pip install -e third_party/Matcha-TTS
pip install modelscope
modelscope download --model FunAudioLLM/CosyVoice2-0.5B --local_dir pretrained_models/FunAudioLLM/CosyVoice2-0.5B
```
已知坑：`monotonic_align` 编译错 → 先 `pip install cython` 再 `pip install -e third_party/Matcha-TTS`；
piper_phonemize wheel 必须保留完整文件名再 pip 装（否则 not a valid wheel filename）。

## 5. 老师对照实验（当天要出的结论）
参考音色素材在本地仓已备好：`out/ref_v1.wav`（7.6s，三条 ASR 验证正确的 V1 体温句）
+ `out/ref_v1.txt`，scp 上云后：
```bash
cd $DATA/matcha-tts-finetune && source .venv/bin/activate
python scripts/gen_teacher_corpus.py --backend cosyvoice \
  --model-dir $DATA/CosyVoice/pretrained_models/FunAudioLLM/CosyVoice2-0.5B \
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
