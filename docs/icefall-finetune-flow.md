# icefall Matcha 微调流程（GPU 机 / 云端）

目标：以基座 V1 pt 热启动，用自录/合成语料微调，降低中文小数/轻声音节错率；
词表、声码器、端侧配置不变，只替换导出的 `model-steps-3.onnx`。

## 0. 环境
- Linux + NVIDIA GPU（24GB 最稳；12GB 可跑但要减 batch）。云可用 AutoDL。
- **icefall 用 fork 分支 `https://github.com/snowlyg/icefall` 的 `matcha-zh-en-16k`**，
  微调必需的 5 处改动已经并进去，clone 完不需要再改一行代码：
  ① `train.py` 不插 blank；② `compute_fbank_baker_zh.py` 22050→16000；
  ③ `convert_text_to_tokens.py` 缺 `normalized_text` 时回退 `.text`；
  ④ `export_onnx.py` 元数据 `sample_rate=16000`；
  ⑤ `export_onnx.py` 加载 ckpt 后用 `--cmvn` 覆写 `model.mel_mean/mel_std`（反归一化守卫，
  见第 5 节那条导致首轮训坏的失效模式）。同一份 diff 也存成 `patches/icefall-baker-zh-16k.patch`，
  上游裸 clone 用 `patch -p1 -N --forward` 打即可（`cloud/bootstrap_gpu.sh` §3 两条路都写了）。
  `git clone` 上游原版仅作为补丁基线；装 `requirements-tts.txt` +
  `piper_phonemize`（`pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html`，
  国内走 gh-proxy 下 wheel、保持规范 wheel 文件名再装）。
- 本仓 `scripts/` 用同一个 venv（pypinyin/onnxruntime/soundfile/kaldi_native_fbank/lhotse/sherpa-onnx）。
- **icefall clone 下来还跑不了**，四项都要补（`cloud/bootstrap_gpu.sh` §3 已固化）：
  `export PYTHONPATH=$ICEFALL:$PYTHONPATH`（配方 `from icefall.utils import ...`）；
  `icefall.utils` 还要 `kaldialign`/`sentencepiece`（只在主 `requirements.txt`，装时**滤掉
  torch 家族**免得拖走钉好的 torch）；`matcha/monotonic_align` 要
  `python setup.py build_ext --inplace`；`k2` 不在 PyPI，轮子在 HF `csukuangfj2/k2`，
  且**按 torch 补丁号硬校验**（torch2.7.1 的轮子跑在 2.7.0 上直接 ImportError）。
- **`source .venv/bin/activate` 和 `export PYTHONPATH` 都是会话级的，ssh 重连即失效**，
  此时 `python` 落回镜像自带的 conda 解释器 → `ModuleNotFoundError: No module named 'k2'`
  （看着像环境没装好，其实只是没激活）。`cloud/env_gpu.sh` 一次 source 解决
  （激活 venv + `REPO/ICEFALL/TTS/PY/PYTHONPATH` + `TZ=Asia/Shanghai` + `PYTHONUNBUFFERED=1`），
  bootstrap §1c 会把它装到 `$DATA` 并挂进 `~/.bashrc`。

## 1. 语料 → 原始 cuts
```
python scripts/prepare_data.py --corpus <录音目录> --out data
# 产出 data/fbank/baker_zh_cuts_{train,valid,test}.jsonl.gz（带 supervisions[0].text）
```
没有真人录音时的替代：`scripts/gen_teacher_corpus.py` 批量合成 + SenseVoice 回读门控
（拒绝采样，只收念对的）。先用 `--backend v1 --limit 30` 自采样练手——实测（2026-10-24，
60 句录制稿）：首试通过 ~70%，但有 8 句重试 8 次全败，全是**系统性丢音节**
（患者李四→患者李、安排→安在），自采样修不了这类错 → 主路必须走老师模型。
主路是 `--backend cosyvoice`（CosyVoice2 zero-shot，`--ref-wav/--ref-text` 用 V1 现网
"温柔女声"参考音色片段定音色）——先小样本对照 ruler 基线验证老师错率 <5% 再放大产语料。
字母行（ICU 等）ASR 假错多，门控放宽、需人耳抽听。

## 2. 词表（钉死作者那份 + 补 `_ 0`）
```
python scripts/make_train_tokens.py --vocab models/vocab_tts.txt --out "$TTS/data/tokens.txt"
# = 作者 2190 token（id 1..2190 一位不动）前面补一行 `_ 0`，仅做 padding；
# 推理从不喂 `_`（V1 无 add_blank/BOS，见 author-model-facts.md）。不要跑 generate_tokens.py。
```
⚠ 训练必须关 intersperse（V1 推理不插 blank，导出物才能回落现网）：
把 `matcha/train.py` prepare_input 里 `intersperse_blank=True` 改为 `False`。

## 3. 前端（写 c.tokens）
官方管线是 **raw cuts → convert_text_to_tokens → fbank → 切分**，不能直接喂我们的分片文件：
```
cd egs/baker_zh/TTS
# 3a) 合并成单一 raw cutset（顺序 train→valid→test，便于第 4 步复刻切分）
python -c "from lhotse import CutSet as C; d='<repo>/data/fbank/'; cs=C.from_file(d+'baker_zh_cuts_train.jsonl.gz')+C.from_file(d+'baker_zh_cuts_valid.jsonl.gz')+C.from_file(d+'baker_zh_cuts_test.jsonl.gz'); cs.to_file('data/manifests/baker_zh_cuts_raw.jsonl.gz')"
# 3b) 脚本读的是 supervision.normalized_text（由 `lhotse prepare baker-zh` 生成），
#     自录/合成语料只有 .text → 改一行回退（lhotse 的 custom 字段不随 to_file 序列化，
#     注入 normalized_text 这条路已实测走不通）
sed -i 's|text = c.supervisions\[0\].normalized_text|text = getattr(c.supervisions[0], "normalized_text", None) or c.supervisions[0].text|' local/convert_text_to_tokens.py
python local/convert_text_to_tokens.py \
  --in-file  data/manifests/baker_zh_cuts_raw.jsonl.gz \
  --out-file data/manifests/baker_zh_cuts.jsonl.gz
```
（jieba + lazy_pinyin(TONE3, tone_sandhi=True) + piper_phonemize，与端侧推理前端同源；
自录转写要先规整：数字转汉字、字母拆读，参考 data/record_script.tsv。）

## 4. 特征 + cmvn（V1 是 16k）
- `local/compute_fbank_baker_zh.py` 里 `sampling_rate=22050`、`.resample(22050)`
  改为 **16000**（f_max 保持 8000=16k 奈奎斯特；n_fft/hop/win/mels 不动：1024/256/1024/80）。
  不改不会报错：lhotse 会把 16k 语料插值上采样，特征域静默偏离 V1。
- 需要 `lilcom`（`LilcomChunkyWriter`）；`convert_text_to_tokens.py` 需要 `jieba`——
  两个都不在 icefall 的 requirements 里。
- 算完特征再切分（数量从第 1 步产物读，别写死）：
  `lhotse subset --last <va+te>` → `--first <va>` / `--last <te>`，train 用 `--first <tr>`。
- `local/compute_fbank_statistics.py` 出自录语料的 cmvn → `data/fbank/cmvn.json`。
- **归一化与反归一化必须是同一对数**（这是首轮微调训坏的根因，见第 5 步）：
  `train.py` 的 `prepare_input` 用 `--cmvn` 文件里的 `mel_mean/mel_std` 归一化训练特征，
  而模型**反归一化用的是自身 buffer**——热启动时那对 buffer 来自 V1 (-2.9522, 5.4468)。
  于是"用我们的 cmvn 训练、用 V1 的 buffer 导出"会把输出 mel 仿射拉歪
  （预测 `(x+6.21)×1.878-2.95` ⇒ 均值 -2.95、std 5.45）。**两种自洽选法**：
  ① 热启动时用 `--mel-mean/--mel-std` 把 buffer 覆写成自算 cmvn（z 保持单位方差，与预训练
  假设最贴近）；② 训练时也喂 V1 那对做 `--cmvn`。首轮两个都没做 → 见下。
  `cmvn.json` 是标量对 `{"fbank_mean","fbank_std"}`，不是 80 维向量。

## 5. 热启动
```
# 注意：基座必须是 pytorch_model_sherpa.bin（与现网 steps-3 ONNX 权重一致；
# pytorch_model.bin 是另一份，303/305 张量不同）
python <repo>/scripts/make_warmstart_ckpt.py \
  --bin models/zh-en/pytorch_model_sherpa.bin \
  --tokens "$TTS/data/tokens.txt" \
  --out matcha/exp-1/epoch-1.pt
# 脚本会剥 model. 前缀、把 emb 2190→2191 行（ids 与作者完全一致）
# （确认要换 mel 域时才追加 --mel-mean/-mel-std，取 cmvn.json 里的 fbank_mean/fbank_std）

# ⚠ 先只跑 1 个 epoch 当门控。别用绝对阈值判 loss（见下方"门控怎么读"），
#   做法是同一份数据再跑一次随机初始化当对照。
python matcha/train.py --exp-dir matcha/exp-1 --start-epoch 2 --num-epochs 2 \
  --max-duration 60 --bucketing-sampler 1 --num-workers 4 --num-buckets 10 --save-every-n 100
python matcha/train.py --exp-dir matcha/exp-1 --start-epoch 2 --num-epochs 302 \
  --max-duration 60 --bucketing-sampler 1 --num-workers 4 --num-buckets 10 --save-every-n 20 --world-size 1
# 12GB 显存：--max-duration 降到 200–400；--num-epochs 默认 1000 是从零训练用的，微调别照抄
```

`train.py` 这几个参数/取日志的坑（实测 2026-10-25）：
- `--num-epochs` 是**最后一个 epoch 的索引（含）**，不是轮数：`--start-epoch 2 --num-epochs 2`
  实测跑了一整轮 epoch 2，`--num-epochs 8` 跑到 `Epoch 8`。跑 N 轮 → `--num-epochs $((start+N-1))`。
  配合下一条读：`--start-epoch N` 会去加载 `epoch-(N-1).pt`。
- 没有 `--log-interval` / `--valid-interval` 参数（写死 10 / 1500）。语料 <1k 条时
  `--max-duration 600` ≈ 每 epoch 6 个 batch → 一行 step loss 都打不出来，
  门控阶段用 `--max-duration 60` 换 ~50 个 batch。
- `--num-buckets` 默认 30，必须 ≤ valid cuts 数，否则 lhotse `estimate_duration_buckets` 直接 assert。
- `data_args.add_blank=True` 是**死配置**：全仓只有 `train.py:149` 那一处字面量，
  没有任何代码读它；真正插 blank 的是 `prepare_input` 的 `intersperse_blank`（见上，已关）。
- 取日志时别在 `| tee x.log | grep ... | head -N` 后面接 `head`：`head` 读满就关管道，
  python 收 SIGPIPE **当场死**，epoch 汇总和 ckpt 落盘都不会发生（看着"跑完退出了"其实没跑完）。
  要tee 到文件、事后单独 `grep` 文件。

### 轮数：19 条验证集判不出早停点，用错率选
实测四个验证点（0.92h 语料、18.6M 参数、恒定 lr 1e-4、`--max-duration 60`）：

| epoch | dur_loss | diff_loss | tot_loss(valid) |
|---|---|---|---|
| 2（=热启动 0 step） | 1.791 | 0.870 | 3.843 |
| 29 | 0.450 | 0.267 | 1.791 |
| 57 | 0.340 | 0.294 | 1.701 |
| 85 | 0.384 | 0.350 | 1.801 |
| 113 | 0.404 | **0.280** | 1.749 |

`tot_loss` 在 1.70–1.80 之间来回跳、`diff_loss` 非单调（0.267→0.294→0.350→**0.280**），
而同期 train 累计 loss 一路降到 1.455。**结论只能是：19 条样本的抖动量级 ≈ 曲线本身的变化量级，
valid 曲线在这份语料上没有早停判别力**（我曾据 3 个点判"57 见底=过拟合"，第 4 点就把趋势推翻了）。
所以做法不是提前停，而是：**每 20 epoch 存一份 ckpt 多留候选 → 各导出 ONNX → 用 ruler 错率选**。
裁判是错率（`ruler.py --n 100`，同一台机器同一套 ASR 才可比），不是 valid loss。
### 门控怎么读：用随机初始化做对照，不要用绝对阈值
`icefall.checkpoint.load_checkpoint` 默认 `strict=False`，键名没对上**不会报错**、
对应模块就静默留在随机权重上；而 `diff_loss` 的量纲取决于"归一化后 mel 的方差"
（mel 被 `(x-mean)/std` 除过，`diff_loss` ≈ `1 + var(mel_normalized)` 才是概率线），
换一个 cmvn 尺度就换一个绝对值 → **任何写死的阈值（如"0.1–0.4"）都不成立**。

对照法（~40 秒，空 exp-dir 即随机初始化）：
```bash
mkdir -p matcha/exp-scratch && python matcha/train.py --exp-dir matcha/exp-scratch \
  --start-epoch 1 --num-epochs 2 --max-duration 60 --bucketing-sampler 1 \
  --num-workers 4 --num-buckets 10 --save-every-n 100 2>&1 | grep "batch 0,"
rm -rf matcha/exp-scratch
```
2026-10-25 实测基线（0.92h 合成语料、936/19/20、V1 mel buffer 原样保留）：

| | `dur_loss` | `prior_loss` | `diff_loss` | `tot_loss` |
|---|---|---|---|---|
| 随机初始化 batch0 | 3.282 | 1.490 | 2.951 | 7.723 |
| 热启动 0 step（epoch 开头 valid） | 1.791 | 1.182 | 0.870 | 3.843 |
| 热启动 batch10 | 1.328 | 1.127 | 0.400 | 2.855 |

三个头**同时**低于随机起点才算热启动生效（只看 `diff_loss` 可能被 mel 域巧合蒙过）。

⚠ **但 loss 健康完全不保证模型可用**——这是首轮花掉一天学到的。ep100 导出后在 ruler 上
**92.2% (461/500)**，基线 V1 同尺子 25.8%；逐条全崩（`二十一床正在呼叫` 3%→99%、
`A;一零床` 5%→100%），错误形态是音节级多插/吞并（`三十六点点五度`、`i i c u`）。
逐项排除后定位到 mel buffer 与 `--cmvn` 不是同一对：候选 ONNX 里含 V1 的 `(-2.9522, 5.4468)`
各 1 个标量、**我们那对 0 个**，而实测输出 mel 统计 `mean=-3.86 std=5.71` 正是被拉歪后的预测值
（V1 同输入是 `mean=-6.08 std=8.27`）。前端与 token 约定都排除了：`lhotse` 的
`batch["tokens"] = [cut.tokens ...]` 走 cut 级 `custom["tokens"]`，实测是音节级
`yong4/yao4/shi2`，与端侧同源。**权重是在"我们那对"的 z 空间里学的，所以不必重训：
用 `cloud/fix_mel_buffer.py` 覆写 ckpt 的 `mel_mean/mel_std` 再导出即可**
（`cloud/session_fix_mel.sh` 一次跑完取证→覆写→导出→校验常数→打包）。第 ⑤ 处补丁（fork 分支 `matcha-zh-en-16k`）之后，这个失效模式在导出环节不可能再发生。
另外：`--start-epoch N` 加载的是 `epoch-(N-1).pt`（日志里 `Loading checkpoint from .../epoch-1.pt`
就是 `--start-epoch 2` 触发的）→ **断点续训要写 `--start-epoch <已存epoch+1>`**。
续训时 optimizer 和 `batch_idx_train` 都随 ckpt 恢复（实测 `global_batch_idx` 从上一轮的 6470
**继续**到 6560，不是从 0 重来）→ `valid_interval` 的计数也是连续的，验证点位置可以据此推算。

## 6. 导出 ONNX
```
python matcha/export_onnx.py --exp-dir matcha/exp-1 --epoch <N> \
  --tokens data/tokens.txt --cmvn data/fbank/cmvn.json
# 产出 model-steps-2..6.onnx；取 steps-3 替换端侧那份
```
⚠ **导出物落在当前工作目录、文件名写死 `model-steps-2..6.onnx`**（不在 `--exp-dir` 里，
实测 `ls matcha/exp-1/*.onnx` 为空），且**换 epoch 再导就原地静默覆盖**——连导 40/60/100 三轮，
只剩最后那批。多候选时必须每导一个就 `cp model-steps-3.onnx cand/steps-3-ep$e.onnx` 存档。
接口实测（`--help`）只有这 4 个参数，没有别的开关；**默认值很危险**：
`--exp-dir` 默认 `matcha/exp-new-3`、`--epoch` 默认 `2000`（help 注明 "Epoch counts from 1"），
不显式传就会去加载不存在的 checkpoint。`--cmvn` 的 help 文案写错了（写成 "Path to vocabulary"），
按参数名给 `data/fbank/cmvn.json` 即可。
需 `torch>=2.2.0`（否则 SDPA 导出报错）。
⚠ 元数据（实测 `matcha/export_onnx.py`）：
- **第 190 行 `"sample_rate": 22050` 必须改成 16000**（现网 V1 即 16000，端侧按它定时长/采样）；
  改法见 `cloud/run_finetune.sh` [6]，是幂等 sed。
- 第 196 行 `"use_eos_bos": 0` **源码不用改**：sherpa-onnx 的 `ApplyBosEos` 对 zh-en 是直通
  （见 author-model-facts），它是惰性键；最终覆盖步骤会把它写成 V1 的 `1`，两边别各自"修正"。
- 导出物里**不要出现 `add_blank` 键**（sherpa-onnx 见到它就会在推理时插 blank）。
- **键集合对不上，逐键手工对齐是错的做法**。实测 icefall 导出 15 键、V1 是 12 键，
  差集不小：导出多 `has_espeak/jieba/dataset/maintainer/dataset_url/dataset_comment`、
  缺 `voice/demo_url/modelscope_url`，`language` 更是从 `chinese English` 变成 `Chinese`、
  `model_author` 从 `dengcunqin` 变成 `icefall`。`language`/`voice` 是端侧前端会读的键，
  靠人判断"哪些影响运行时"迟早漏 → **把 V1 的 `metadata_props` 整份覆盖到导出物上**，
  diff 空集是构造性保证（工具 `scripts/fix_onnx_meta.py`，本地不入库，已在
  ref=src=V1 与"15 键导出物"两种输入上自测通过）：
```bash
python scripts/fix_onnx_meta.py --ref models/zh-en/model-steps-3.onnx \
  --src <导出的 model-steps-3.onnx> --out model-steps-3.fixed.onnx
```
  默认 `--keep num_ode_steps`：该键描述这张图实际跑了几步 ODE，跟着参照走会让
  `model-steps-5.onnx` 的元数据说谎；只有 steps-3 那份能整份照抄。
- 覆盖后仍要逐键 diff 一遍（清单在 author-model-facts）：
```bash
.venv/bin/python -c "
import onnx
for f in ['<新导出>/model-steps-3.onnx', 'models/zh-en/model-steps-3.onnx']:
    d = {p.key: p.value for p in onnx.load(f).metadata_props}
    print(f, len(d)); print(sorted(d.items()))
"
```

## 7. 验收
⚠ **尺子的短语必须与训练语料不相交**，否则测的是"背题"。首轮就踩了：语料和尺子都取自
同一份生产短语表，`体温三十六点五度` 在 train 切分里逐字出现 **3 次**，微调后该条从 45%→0%
——看着像"小数问题治好"，其实是记忆；同时不在语料里的 `二十一床正在呼叫` 从 3%→100%、
`A;一零床正在呼叫` 5%→95%（`一零` 在语料里 0 句）。**正确做法**：把已知难例只留在尺子里，
教它的时候换表面形式（同一模式换数字/换主语，如 37.2 度、38.4 度），
并在生成语料前先把尺子短语从候选表里剔除、事后用 grep 复验零交集。
验收尺子在 `scripts/ruler_heldout.tsv`（28 条，与 `out/corpus1k` 975 句**字符整句、子串、
拼音同句**三种命中均为 0，2026-10-25 复验）。

⚠ **同音字假错**：ASR 写的是汉字不是读音，`输液` 会被写成 `书叶/树叶/书页`，字符口径把它
记成真错。所以 `ruler.py` 同时报两个口径（`asr.py:py` 做拼音比较）：
**字**＝严格字符、**音**＝只看读音。V1 基线两者差 ~6pp（30.0% vs 24.3%）＝度量噪声量级；
如果候选只涨字口径是噪声，**两个口径一起涨才是真损伤**。

## 7.1 首轮微调的 held-out 实测结论（2026-10-25，n=30/条，同机可比）

| 模型 | 总计(字/音) | 小数/剂量 | 时间 | 床号 | 轻声/儿化 | 多音字 | top 变体少音节 |
|---|---|---|---|---|---|---|---|
| V1 基线 | 30.0%/24.3% | 12/12 | 32/32 | 23/16 | 19/14 | 45/45 | **0/25** |
| ep60 | 83.2%/83.1% | 79/79 | 53/52 | 99/99 | 78/78 | 100/100 | 11/25 |
| ep180 | 86.7%/85.5% | 64/64 | 98/89 | 89/89 | 92/92 | 100/100 | 14/25 |
| ep300 | 84.3%/83.2% | 70/70 | 96/93 | 82/82 | 85/85 | 100/100 | 14/25 |

原始日志 `out/heldout2/*.log`，汇总 `out/heldout2/summary.md`（三份都是 mel 覆写后的导出物，
即上一节那个域 bug 已修）。

结论三条，都还有证据支撑、不是推断：
1. **这套配方（0.92h 教师语料 + 全参数 lr 1e-4）在没见过的句子上净负收益**：两个口径一起
   从 ~25% 涨到 ~84%，不是度量噪声。之前"体温 45%→0%"的收益全部是被背题撑起来的。
2. **损伤形态是吞字/时长塌，不是读错音**：25 条中文短语里，V1 的 top 变体 0 条比期望少音节，
   ep60 有 11 条、ep180/300 有 14 条。例：`一零三床准备接诊`→"一零三床准备枕"、
   `零点二五毫克每日两次`→"零点二五每日日每四"、`他刚走了一会儿`→"他刚走一会儿"。
   → 病灶在 **duration 头/对齐**，与教师语料的节奏（短句、吞 了/儿）一致。
3. **V1 自己的弱点在 held-out 上换了位置**：整段"小数/剂量"类只有 12%，但
   `血氧饱和度九十七点三` 93%、`早晨六点四十分抽血` 87%（"六点四四十分"＝数字重复），
   `九床的…` 100% 里含同音假错。**要治的是"数字串重复/吞字"，不是"小数读不出"**。

所以**先别再投训练**：下一轮改配方（降 lr / 冻结 CFM decoder 只留 encoder+dur / 语料按数字空间
系统覆盖 / 已知难例绝不进语料），改完先跑 40~50 轮短训 + `ruler_heldout.tsv` 门控，
错率低于 V1 才续长训。判据一律用 held-out 音口径，不用 valid loss（§5 已证 19 条验证集判不出东西）。

- 把导出的 `model-steps-3.onnx` 放进 `models/zh-en/`，跑 `scripts/ruler.py --n 100`，
- 把导出的 `model-steps-3.onnx` 放进 `models/zh-en/`，跑
  `scripts/ruler.py --n 100 --phrases-file scripts/ruler_heldout.tsv`（**必须带 held-out 集**，
  默认那 5 条已在训练语料里，只用来确认"没训坏"），对比微调前后错率。
  达标线：**音口径**中文各条 <5%、床号/字母不回归、RTF 不明显变差。
- 通过后替换端侧 `model-steps-3.onnx` 并清模型缓存验收（见 edge-deploy-constraints.md）。
