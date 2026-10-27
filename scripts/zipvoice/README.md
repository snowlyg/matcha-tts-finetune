# ZipVoice 端侧评估（已判死，留的是尺子）

一句话结论：**ZipVoice 在我们这批设备上端侧彻底出局，而且是实测不是外推**（RTF 差约 17 倍）。
以后有人再提"换 ZipVoice / CosyVoice 这类新基座"，先引这组同机数字，别再重跑一遍。
要区分两件事：**准确率优势是真的**（属训练侧参考），**端侧可行性是否掉的**。

## 真机实测（现网那台 8 核、32 位软浮主机，threads=4、steps=4）

| | ZipVoice int8 | Matcha V1（现网） |
|---|---|---|
| 逐句 RTF | 33.3 / 34.1 / 41.2 / 35.1 / 46.7 | 1.09–1.42，中位 **1.20**（n=20） |
| 一条话术耗时 | **68–74 秒**（音频才 1.5–2.2 秒） | 1.7–4.1 秒 |
| 模型加载 | create_ms **31,047** | 11–16 秒 |
| 峰值 RSS | **422 MB** | 294 MB |

判据 RTF ≤ 2 → 差约 17 倍；同机比值 **~29×**。PC 同机对照（i5-11500 / 4 线程 / 同 20 句 / sherpa 1.13.8）
方向一致：Matcha 中位 0.082，ZipVoice steps=4 中位 4.082（**49.5×**）、steps=10 中位 9.240（112×）。
**steps 是流匹配采样步数，耗时随它近似线性走**——报 ZipVoice 的 RTF 不写步数等于没写。

准确率对照（ZipVoice 半边）**没测完就停了**：PC 那把尺子只跑出 Matcha 一边
（14 句 × 8 次，严格字符 61/112=54.5%、去调拼音 52/112=46.4%）。这两个口径比
"体温 50% / 自动接听 42%" 那批更宽，别混用。所以"ZipVoice 更准"目前只是 Mac 上的观察，本仓未复现。

## 两个版本边界（决定"能不能直接换"）

- 现装 runtime（1.12.18 + 自研补丁）**加载不了这个 int8 中英包**：我们这版 zipvoice 走 cppinyin +
  `--zipvoice-pinyin-dict`，而包内只有三列 `lexicon.txt`（`一 y0 i1`），cppinyin v0.10 要"词 词频 音节"
  → 设备 `EXIT=255`。上游 `Remove cppinyin from zipvoice (#2892)` 的首个 tag 是 **v1.12.20**，要用这个包就得 ≥v1.12.20。
- 上游"中英混读要 ≥v1.12.19（英文分词修复 #2834）"这条**对我们不构成阻塞**：那个 commit 已在 1.12.18 树内
  （`git merge-base --is-ancestor` 为真）。判版本边界一律用 `git tag --contains`，别拿 release 说明当准。
- v2.0.0 移除了 espeak-ng/piper-phonemize（#4055），`--zipvoice-data-dir` 已废弃并被忽略；
  英文读音要另外喂 **`lexicon-en-us.txt`**（约 6.3 MB，在 tts-models release 下）。
  声码器 `vocos_24khz.onnx`（约 54 MB）也是独立资产，不在模型包里。
- **上游确实发 OHOS armv7 预编译包而且是软浮**（v1.12.20 / v2.0.0 都有
  `sherpa-onnx-v<ver>-ohos-armeabi-v7a.tar.bz2`，`readelf -A` 只有 `Tag_FP_arch: VFPv3`、无 `Tag_ABI_VFP_args`）。
  以前"上游不发 OHOS 预编译"的说法是错的。但包里只有 `.so` + `.pc`，**没有头文件也没有 CLI**，
  所以要测必须自己写 C 驱动交叉编译——这就是本目录存在的原因。
- 接进 App 不是"换一个模型文件"：v2.0.0 要新增 `SherpaOnnxOfflineTtsGenerateWithConfig` 等符号、
  新配置字段、随包参考音频与它的转写文本，我们的 shim 需要重编。

## 脚本顺序

```bash
bash scripts/zipvoice/probe_api.sh                       # 先看目标 tag 的 C API 形状（结构体字段随版本变）
SHERPA_PKG=<ohos 包> bash scripts/zipvoice/check_abi.sh   # 确认预编译包是软浮、能上这台设备
OHOS_LLVM=... OHOS_SYSROOT=... SHERPA_PKG=... bash scripts/zipvoice/build_ohos.sh   # Linux 侧交叉编译
python scripts/zipvoice/pc_same_machine_rtf.py            # PC 同机对照（先只出 Matcha 分母）
# 设备上：把 edge_baseline.sh / zipvoice_bench / phrases.txt 与模型推到同一个工作目录
sh edge_baseline.sh                                       # 同机分母（含峰值 RSS 轮询）
./bin/zipvoice_bench model matcha/vocos lexicon-en-us.txt phrases.txt 4 4
```

## 踩过的坑（都会让人误判成"模型不行"）

- **C API 结构体没有默认值**：`memset` 之后 `feat_scale`/`target_rms` 是 0，`Validate` 直接失败——
  必须显式赋值（`zipvoice_bench.c` 里那四行就是干这个的）。
- **官方库在 OHOS 把日志写 hilog**（tag `A06666/sherpa_onnx`），stderr 是空的；要先 `hilog -r` 再 `hilog -x` 抓。
- **`hdc file send` 给一个不存在的目标名会把它建成目录**，`ls -l` 输出以 `total` 开头就是中招了。
- Windows 侧 SDK 没有 ELF clang，交叉编译只能在 Linux 上做。
