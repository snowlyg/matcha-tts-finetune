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
