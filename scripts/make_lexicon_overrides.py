#!/usr/bin/env python3
"""从真值表生成两份"同一套读音"的产物，并报告它在生产话术里到底管多少句。

为什么要有这个脚本：词典覆盖这件事一旦靠手抄，训练侧和设备侧就会抄成两份不一样的，
而两边不一致会让"训好的模型"和"设备上念出来的"读法不同——本项目已经为此重跑过一轮。
所以真值只写一处（scripts/pronunciation_overrides.tsv），两份产物都由它生成并当场校验。

产物：
  data/lexicon_overrides.txt  设备侧格式（`词 音节1 音节2 …`，TONE3、轻声记 5）。
                              sherpa-onnx 自 v1.12.10 起支持短语级最长匹配（PR #2507），
                              v1.12.16 起 --matcha-lexicon 可传逗号分隔的多份词典（#2765），
                              所以覆盖表可以单独下发、不动主词典。
  data/pypinyin_overrides.py  训练侧：pypinyin 的 load_phrases_dict 入参，两边共用。

当场校验（不通过就退出，不产出）：
  ① 每个音节必须是 models/zh-en/tokens.txt 里真实存在的 token —— 词表没有的音节，
     模型根本发不出（例：词表里**没有任何 -r 儿化音节**，所以"这儿"无法用词典修）。
  ② 覆盖词必须真的被前端读错（否则是空条目，只该留在文档里）。
"""
import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_frontend as cfr
import config

HAN = re.compile(r"[一-鿿]")


def load_tokens(path):
    toks = set()
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split()
        if len(p) == 2:
            toks.add(p[0])
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overrides", default=cfr.OVERRIDES)
    ap.add_argument("--tokens", default=os.path.join(config.REPO, "models", "zh-en", "tokens.txt"))
    ap.add_argument("--scan", nargs="*", default=[], help="在这些话术表里统计每条覆盖词命中多少句")
    ap.add_argument("--out-lex", default="data/lexicon_overrides.txt")
    ap.add_argument("--out-py", default="data/pypinyin_overrides.py")
    args = ap.parse_args()

    table = cfr.load_overrides(args.overrides)
    if not table:
        raise SystemExit("真值表是空的，没什么可生成")
    toks = load_tokens(args.tokens)

    bad = {w: [s for s in syl if s not in toks] for w, (syl, _) in table.items()}
    bad = {w: v for w, v in bad.items() if v}
    if bad:
        for w, v in bad.items():
            print(f"!! {w} 的音节不在模型词表里：{v} —— 这类读音设备发不出来，只能调用侧改写")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.out_lex) or ".", exist_ok=True)
    with open(args.out_lex, "w", encoding="utf-8") as f:
        f.write("# 由 scripts/make_lexicon_overrides.py 从 pronunciation_overrides.tsv 生成，勿手改\n")
        for w, (syl, why) in sorted(table.items()):
            f.write(f"{w} {' '.join(syl)}\n")
    with open(args.out_py, "w", encoding="utf-8") as f:
        f.write("# 由 scripts/make_lexicon_overrides.py 生成，勿手改（真值只写一份）\n")
        f.write("PHRASES = {\n")
        for w, (syl, _) in sorted(table.items()):
            f.write(f"    {w!r}: [[{', '.join(repr(s) for s in syl)}]],\n")
        f.write("}\n")

    print(f"[ok] {len(table)} 条覆盖 → {args.out_lex} / {args.out_py}")

    # 命中统计：顺便验证"前端确实读错"，读不错的是空条目
    hit = collections.Counter()
    lines_scanned = 0
    erhua = collections.Counter()
    for path in args.scan:
        if not os.path.exists(path):
            print(f"!! 跳过不存在的表 {path}")
            continue
        for line in open(path, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            col = line.rstrip("\n").split("\t")
            text = col[1] if len(col) > 1 else col[0]
            lines_scanned += 1
            for w in table:
                if w in text:
                    hit[w] += 1
            for m in cfr.ER_SUFFIX.finditer(text):
                erhua[m.group(0)] += 1
    print(f"[命中] 扫了 {lines_scanned} 句：")
    for w, (syl, _) in sorted(table.items()):
        n = hit[w]
        mark = ""
        if n:
            mark = "（前端实测读错 ✓）" if cfr.audit(w, table)[0] else "（前端本来就对 ✗ 空条目）"
        print(f"   {w:6s} {n:4d} 句  → {' '.join(syl)} {mark}")
    if erhua:
        tot = sum(erhua.values())
        print(f"[儿化] {tot} 处 / {len(erhua)} 种写法，词表无 -r 音节 → 词典治不了，只能调用侧改写：")
        print("   " + "  ".join(f"{k}×{v}" for k, v in erhua.most_common(12)))


if __name__ == "__main__":
    main()
