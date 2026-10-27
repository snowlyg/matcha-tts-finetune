# 端侧取证台架（在真机上跑，不是训练链路）

为什么这些脚本进仓库而不是留在本机：本仓所有错率本来是"Mac 上合成 + Mac 上 ASR 回读"，
而设备上的 sherpa 前端在**含标点的句子**上行为不等价（`。 ， 、 ！ ？` 会把一条播报拆成多次独立推理，
`； ;` 不拆，空格和换行连 token 都不产生）。所以"改书面边界能降错率"这类结论**只能在设备上重测**，
取证方法必须能重跑。结论本身见 [../docs/device-runtime-check.md](../../docs/device-runtime-check.md)。

## 前置（一次性，脚本不代做）

设备上 `$TTS_DEV_DIR`（默认 `/data/local/tmp/tts-probe`）里要有：

- `bin/sherpa-onnx-offline-tts` 和 `lib/`（与生产同一版本的 OHOS 构建，softfp armv7）
- 模型目录 `matcha/`：`model-steps-3.onnx`、`vocos-16khz-univ.onnx`、`lexicon.txt`、`tokens.txt`、
  `espeak-ng-data/`、`number-zh.fst`、`date-zh.fst`、`phone-zh.fst`

**红线**：只在 `$TTS_DEV_DIR` 里写，绝不覆盖生产的 rawfile / filesDir 模型文件。

## 三个入口

```bash
# 1) 标点存活矩阵：同一条话术换 12 种写法，各合成一次
#    报 token 行数 / 推理次数 / 音频时长 / 合成耗时
TTS_TARGET=<设备> MODE=matrix bash scripts/device/pull_and_manifest.sh

# 2) 同句 N 次 A/B：成对的 a/b 档**交错**跑 8 轮（设备漂移对两档同等）
TTS_TARGET=<设备> MODE=ab CASES=data/punct_cases.tsv bash scripts/device/pull_and_manifest.sh

# 3) 拉回来的样音：客观停顿 + 回读打分
python scripts/device/measure_pause.py --dir out/device/punct
python scripts/score_wavs.py --manifest out/device/punct/manifest.tsv
```

## 口径与坑

- **停顿**：10ms 帧、帧 RMS、阈值 `max(1e-5, 0.02×全句峰值)`；裁首尾静音得语音段，语音段内连续低于阈值
  的最长一段 = 句内最长静音。别把"整段时长"当停顿——首尾静音也计在整段里。
- **两个时长要分清**：加边界让**音频**多 0.29~0.43 秒，同时因为拆成两次推理让**合成耗时**涨 35%~53%
  （+0.98~1.19 秒；RTF 1.08→1.35，仍在 ≤2 预算内）。连播排队的压力看后者，听完的长短看前者。
  别把这写成"翻倍"——那是把绝对增量当成倍数。
- **TTS 的开关叫 `--tts-rule-fsts`**，`--rule-fsts` 是 ASR 的；给 TTS CLI 传后者会走"未知选项"分支
  `exit(-1)`（RC=255），曾被误判成"带 fst 会崩"。
- **token 级证据只在 hilog**（tag `sherpa_onnx`），CLI 的 stderr 只有 config 回显。
- **设备侧只能用 POSIX sh**：设备没有 `expr`，`while [ $I -le N ]` + `expr` 会静默只跑一轮；
  重复轮次靠 `REPS_LIST` 传一个列表进来。
- **n=8 只够判"有没有新增错误类型"**（比如拆段带来的叠字），不够报百分比；这模型同参数每次结果都不同。
- `hdc file send/recv` 给一个不存在的目标名会**建成目录**，所以先 `mkdir -p` 再传，拉完用 `ls` 确认是文件。
