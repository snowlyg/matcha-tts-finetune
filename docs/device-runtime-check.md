# 端侧运行时取证（开源鸿蒙真机）

要查的东西只有三个：**那份 sherpa-onnx 是什么版本**、**它带的 `lexicon.txt`/`tokens.txt`/`*.fst`
长什么样**、**实际 ABI 是什么**。这三样决定了两件我们真正想做的事：

- 下发**短语发音覆盖词典**（需 ≥ v1.12.10 的短语最长匹配 #2507；要多份词典合并需 ≥ v1.12.16 #2765）；
- 将来若换模型族，运行时支不支持（例：ZipVoice 需 ≥ v1.12.11，已因 RTF 判死，不用再查）。

> **工具是 `hdc` 不是 `adb`**：真机是 OpenHarmony，产物是 `.hap` + native `.so`，没有 APK。
> `hdc` 随 DevEco Studio / OpenHarmony SDK 的 `toolchains` 目录分发；`hdc -v`、`hdc list targets` 先确认连上了。
>
> 下面的路径**不要照抄**——不同系统形态（标准系统 / 轻量系统）差别很大，所以每一步都写成
> "先探再取"：先 `find`，把探到的实际路径填进 `hdc file recv`。

## 0. 先确认系统形态与 ABI（一条命令）

```sh
hdc shell 'param get const.system.version; param get const.product.model;
           param get const.product.cpu.abilist; getconf LONG_BIT; bm dump -a 2>&1 | head -3'
```

- `bm dump -a` 报 `command not found` → 不是标准系统（没有 HAP 包管理），
  那 sherpa 多半是**静态链进应用可执行文件**或放在厂商分区里，第 2 步要按可执行文件去 `strings`。
- `abilist` 若显示 `arm64-v8a`，则本仓文档里"32 位 armeabi-v7a"的记录需要当场更正。

## 1. 最省事的一条路：从构建产物取，而不是从设备取

有 App 工程源码时，版本写在依赖里、`.so` 就在 libs 目录，一条 grep 完事：

```sh
grep -rn "sherpa" oh-package.json5 entry/oh-package.json5 2>/dev/null
find . -name "libsherpa*" -o -name "libonnxruntime*" | head
strings -a entry/libs/*/libsherpa_onnx.so | grep -oiE "sherpa-onnx-v?[0-9]+\.[0-9]+\.[0-9]+" | sort -u
```

## 2. 没有工程源码：在设备上把 .so 搜出来

```sh
hdc shell 'find /data /vendor /system /system_ext /mnt -xdev \
             \( -iname "*sherpa*" -o -iname "*onnxruntime*" \) 2>/dev/null | head -30'
```

取回到 Mac 上判版本与特性（设备上一般没有 `strings`/`nm`）：

```sh
hdc file recv <第 2 步探到的路径> /tmp/device_lib.so
strings -a /tmp/device_lib.so | grep -oiE "sherpa-onnx-v?[0-9]+\.[0-9]+\.[0-9]+" | sort -u | head
```

## 3. 版本字符串查不到时，用"特性探针"定区间

版本常量常被优化掉，但**符号与错误提示字符串会留下年代痕迹**：

```sh
strings -a /tmp/device_lib.so | grep -ciE "jieba"        # >0：早于 v1.12.15（#2664 移除了 cppjieba）
strings -a /tmp/device_lib.so | grep -ciE "phrase"       # >0：有短语匹配（#2507 ≥1.12.10 或 #2668 ≥1.12.15）
strings -a /tmp/device_lib.so | grep -ciE "zipvoice"     # >0：≥ v1.12.11
strings -a /tmp/device_lib.so | grep -ciE "rule_fsts|phone.fst|number.fst"
nm -D /tmp/device_lib.so 2>/dev/null | grep -iE "Zipvoice|PhraseMatcher|JiebaLexicon|Matcha" | head
```

组合起来能夹出区间：**有 `jieba`、有 `phrase`、无 `zipvoice` → 大致落在 v1.12.10 ~ v1.12.14**。
这正是"能不能下发短语覆盖词典"的关键分界（≥1.12.10 才能）。

## 4. 取模型资产（写覆盖词典前必须先看一眼）

```sh
hdc shell 'find /data/app /data/service /vendor -xdev \
             \( -iname "lexicon.txt" -o -iname "tokens.txt" -o -iname "*.fst" -o -iname "*.far" \) 2>/dev/null | head -20'
hdc file recv <探到的目录>/* /tmp/device_assets/
```

拿到后本地核对三件事（这三条决定覆盖词典能不能生效）：

```sh
wc -l /tmp/device_assets/lexicon.txt                  # 行数：主词典规模
awk "NF>2" /tmp/device_assets/lexicon.txt | head       # 有没有**多音节行**（短语行的证据）
head -3 /tmp/device_assets/tokens.txt; wc -l < /tmp/device_assets/tokens.txt
diff <(cut -d" " -f1 /tmp/device_assets/tokens.txt) \
     <(cut -d" " -f1 models/zh-en/tokens.txt) | head   # 设备词表 vs 我们训练词表，必须逐行相同
```

**注意权限**：非 root 真机上应用沙箱（`/data/app/el2/100/base/<bundle>/…`）通常读不到。
读不到就让 App 侧把模型目录导出来（或走工程产物那条路，第 1 节），不要试图改设备分区。

## 5. 换模型文件后的缓存

应用会把模型解包到沙箱里，**只替换包内文件不会生效**，必须清掉沙箱那份缓存目录再重启应用。
确切目录由第 4 步的 `find` 结果决定（别按 Android 的 `filesDir` 猜）。

## 已取证结果（2026-10-26，**真机**，非 WSL）

| 项 | 实测 |
|---|---|
| 运行时版本 | `libsherpa-onnx-c-api.so` 的 strings → **1.12.18** |
| 生产模型目录 | `/data/app/el2/100/base/<bundleName>/haps/<moduleName>/files/matcha-icefall-zh-en/`（真机上 `find` 探得；包名/模块名属项目内部信息，不入库） |
| 目录内容 | `model-steps-3.onnx`、`vocos-16khz-univ.onnx`、`lexicon.txt`、`tokens.txt`、`number-zh.fst`、`date-zh.fst`、`phone-zh.fst`、`espeak-ng-data/`；**没有 `dict/`** |
| 主词典规模 | `lexicon.txt` **68,037 行**，其中 **47,113 行是多音节短语**（`十分 shi2 fen1`、`监护 jian1 hu4`） |
| 词表同源性 | 设备 `tokens.txt` 与我们训练用那份 **md5 相同**（`7b271c59cf99530f096b9a02382bac3c`，2190 行）；第 1 行是空格，`;`=2 `:`=3 `,`=4 **`.`=5** `!`=6 `?`=7 |

**由此确定的三件事：**

1. **短语级发音覆盖可以下发**：1.12.18 ≥ v1.12.10（#2507 短语最长匹配）且 ≥ v1.12.16（#2765 `--matcha-lexicon` 可传逗号分隔多份词典），
   所以 `data/lexicon_overrides.txt` 能作为**单独一份覆盖词典**下发，不动那 68k 行主词典。
2. **设备侧确实有 TN**：`number-zh.fst / date-zh.fst / phone-zh.fst` 都在。所以"直接喂阿拉伯数字让 FST 去念"
   是一条**设备上可行、但我们本机测不了**的路（本仓复刻前端不做 TN，`asr.py("138床")` 会把 `138` 原样吐出来）。
3. **热启动的前提被证实**：设备词表与训练词表逐字节相同 → `make_train_tokens.py` 不重排 id 这条纪律是对的。

**⚠ 一条会动摇既有结论的推论（重要）：设备上大概已经没有 jieba 了。**
上游在 **v1.12.15** 就把 jieba 从 matcha/kokoro 前端移除（#2662、#2664），改用
`phrase-matcher.cc` 对 `lexicon.txt` 的**最长匹配**（#2668）；设备是 1.12.18，且目录里**没有 `dict/`**
（`dict/` 正是 jieba 的词典目录）——两条证据互相咬合。含义：

- 本仓 `scripts/g2p.py` / `check_frontend.py` 是 **jieba + pypinyin 的复刻**，
  **不是设备上那个算法**。所以"生产话术里前端只读错 5 句（0.19%）、多音字全部读对"
  这个结论**不能当设备结论**——设备上的多音字读法来自那 47k 行短语词典，而不是运行时 pypinyin。
- 好消息是这变得**可查**了：`大夫` 到底读什么，直接查设备词典即可，不用推测：
  ```sh
  grep -n "^大夫[ 	]" /data/app/el2/100/base/<bundleName>/haps/<moduleName>/files/matcha-icefall-zh-en/lexicon.txt
  grep -c "^大夫" <同上>        # 0 行 ⇒ 设备靠逐字回退，读 da4 fu1 是真的；有行 ⇒ 看它给的是什么
  ```
  顺手把 `这儿 / 一会儿 / 事儿` 也查了（词表无儿化音节，若词典里给了 `zhe4 er2` 之类就能印证）。
- 廉价自证：`strings -a libsherpa-onnx-c-api.so | grep -ci jieba` 应为 0、`grep -ci phrase` 应 >0。

**还缺的那一项（新主线的生死题）**：**标点存活矩阵**——`。` 在 1.12.18 的前端里
是变成 token `.`（id 5）、被丢掉、还是触发按句末拆成两次合成。本机的复刻前端会把它送进序列，
但那是复刻，不算证据。见任务 #28。

## 标点存活矩阵（2026-10-26 **真机**实测，同一颗 CLI + 同一份模型，threads=4）

基线句 `中午二点五十分心电监护`（token 映射 9 行 = 11 个 id，因 `十分`/`监护` 各是一行两 id）：

| 边界 | 推理次数(callbacks) | token 行 | 整段 s | elapsed s | 判读 |
|---|---|---|---|---|---|
| 无 | 1 | 9 | 2.384 | 2.657 | 基准 |
| `。` `，` `、` `！` `？` | **2** | 10 | 2.69~2.81 | 3.58~3.73 | **存活但触发拆段**：变成两次独立合成，计算 +39% |
| `；` / ASCII `;` | **1** | 10 | 2.505~2.507 | 2.893~2.938 | **存活且不拆段** ⇒ 与 Mac 复刻行为等价的那个条件 |
| 空格 / 双空格 / `\n` | 1 | 9 | 2.386~2.389 | 2.656~2.661 | token 数不变 ⇒ **完全无效** |
| 句末 `。` | 1 | 10 | 2.407 | 2.681 | 句末标点不拆段 |

根因在代码里：`text-utils.cc:824` 的 `IsPunct` 集合含 `, . ! ? : ， 。 ！ ？` 但**不含 `;`**；
而 App 侧 `sherpa_tts_napi.cpp:299` 写死 `max_num_sentences = 1` ⇒ 中间任一标点必拆两段。

**⚠ 由此产生一条硬约束：Mac 上量到的收益不能直接搬到设备。**
本仓复刻前端（`scripts/g2p.py`）不拆段，`。` 只是一个进序列的 `.` token；而设备上 `。` 会让
**两次独立生成**发生。所以"加句号 → 错率 24.2%→3.6~4.8%"这个结论，在设备上对应的等价条件是
**`;`（单趟 + 序列内边界）**，不是 `。`。用 `。` 得到的是另一种机制（短句 + 跨段无协同韵律），
**必须单独测**，不能引用 Mac 的数。

**两个待测风险**：① `;` 虽然进模型，但 Baker 语料里 `;` 极少，模型可能不知道拿它怎么办（停顿位置怪异）；
② 拆段版计算 +39%，端侧 V1 RTF≈1.3 → 约 1.8，**仍在 ≤2 预算内但余量从 0.7 掉到 0.2**，长句可能撞线。
测法见 `scripts/score_wavs.py`（设备合成 → 回传 wav → 用同一把 ASR 与形态计数打分）。
