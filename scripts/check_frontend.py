#!/usr/bin/env python3
"""审计"设备前端给的读音"与"标准读音"是否一致——这类错**微调治不了**。

为什么要有这个工具：端侧只替换声学 ONNX，文本→token 的前端（jieba+pypinyin 那一套）
是 V1 自带的、改不动。前端一旦把 `大夫` 转成 `da4 fu1`（该读 `dai4 fu5`），模型只是
照着念，**而尺子看不见**：`ruler.py` 的音口径把期望文本也用同一个 pypinyin 现算，
两边一起错就记成 OK。所以先把"前端错读"单列出来，才能把模型的病和前端的病分开归因。

两级结论：
  已裁决 —— pronunciation_overrides.tsv 命中的词，前端输出 ≠ 真值（可当门控，退出码 1）；
  待裁决 —— 结构可疑（`儿` 被拆成独立 er 音节、`一` 的变调/轻声没体现），需人判，不置失败。

用法：
  python scripts/check_frontend.py --tsv scripts/ruler_heldout.tsv --col 2
  python scripts/check_frontend.py --tsv out/corpus1k/transcripts.tsv --col 2
  python scripts/check_frontend.py --text "他是大夫我是护士"
"""
import argparse
import collections
import os
import re

import g2p
from pypinyin import lazy_pinyin, Style

HERE = os.path.dirname(os.path.abspath(__file__))
OVERRIDES = os.path.join(HERE, "pronunciation_overrides.tsv")
HAN = re.compile(r"[\u4e00-\u9fff]+")
ER_SUFFIX = re.compile(r"[\u4e00-\u9fff]儿")


def load_overrides(path=OVERRIDES):
    """词 -> (真值音节 tuple, 说明)。`#` 开头行忽略。"""
    table = {}
    if not os.path.exists(path):
        return table
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) >= 2:
            table[p[0].strip()] = (tuple(p[1].split()), p[2] if len(p) > 2 else "")
    return table


def _run_truth(run, table):
    """一个中文段的标准读音：整段过 pypinyin（词库会定多音字），再用真值表按最长匹配覆盖。

    覆盖允许改变音节数（`这儿`→1 个音节 zhe4r），所以先标记被覆盖的字位，再顺序拼输出。
    """
    base = list(lazy_pinyin(run, style=Style.TONE3, neutral_tone_with_five=True,
                            tone_sandhi=True, errors="default"))
    owner = [None] * len(run)          # 字位 -> 覆盖它的词起点
    for word in sorted(table, key=len, reverse=True):
        pos = run.find(word)
        while pos >= 0:
            if not any(owner[pos:pos + len(word)]):
                for k in range(pos, pos + len(word)):
                    owner[k] = pos
            pos = run.find(word, pos + 1)
    out, i = [], 0
    while i < len(run):
        if owner[i] == i:
            word = next(w for w in table if run.startswith(w, i))
            out.extend(table[word][0])
            i += len(word)
        else:
            out.append(base[i] if i < len(base) else run[i])
            i += 1
    return out


def truth_syllables(text, table):
    out = []
    for run in HAN.findall(text):
        out.extend(_run_truth(run, table))
    return tuple(out)


def frontend_syllables(text):
    """设备前端实际喂给模型的音节（只取中文段；字母段走 IPA，不在本口径内）。"""
    out = []
    for run in HAN.findall(text):
        out.extend(g2p.chinese_to_pinyin(run))
    return tuple(out)


def audit(text, table):
    """返回 (已裁决差异 [(位置, 前端, 真值)], 命中的表内词, 待裁决可疑结构)。"""
    front, true = frontend_syllables(text), truth_syllables(text, table)
    diff = []
    for i in range(max(len(front), len(true))):
        a = front[i] if i < len(front) else "—"
        b = true[i] if i < len(true) else "—"
        if a != b:
            diff.append((i + 1, a, b))
    hits = [w for w in table if w in text] if diff else []
    suspect = []
    if ER_SUFFIX.search(text):
        suspect.append("儿化拆成独立 er 音节")
    if re.search(r"(.)一\1", text) or re.search(r"一[\u4e00-\u9fff]?儿", text):
        suspect.append("一 的变调/轻声未体现")
    return diff, hits, suspect


def rows_from_tsv(path, col):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) >= col:
            rows.append((c[0], c[col - 1]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="")
    ap.add_argument("--tsv", default="")
    ap.add_argument("--col", type=int, default=2, help="文本列（1 起）")
    ap.add_argument("--list", action="store_true", help="打印真值表后退出")
    ap.add_argument("--show", type=int, default=8, help="打印前 N 条命中")
    args = ap.parse_args()

    table = load_overrides()
    if args.list:
        for w, (syl, note) in sorted(table.items()):
            print(f"{w}\t{' '.join(syl)}\t{note}")
        return

    rows = [("<单条>", args.text)] if args.text else (
        rows_from_tsv(args.tsv, args.col) if args.tsv else
        (_ for _ in ()).throw(SystemExit("给 --text 或 --tsv")))

    n_hit = n_susp = shown = 0
    word_cnt, susp_cnt = collections.Counter(), collections.Counter()
    for rid, text in rows:
        diff, hits, suspect = audit(text, table)
        if diff:
            n_hit += 1
            for w in hits:
                word_cnt[w] += 1
            if shown < args.show:
                shown += 1
                print(f"[前端≠真值] {rid}  {text}")
                print(f"          前端={' '.join(frontend_syllables(text))}")
                print(f"          真值={' '.join(truth_syllables(text, table))}")
                print(f"          差={diff[:6]}  命中={hits}")
        if suspect:
            n_susp += 1
            for s in suspect:
                susp_cnt[s] += 1
    print(f"\n共 {len(rows)} 句：已裁决前端错读 {n_hit} 句、待裁决可疑 {n_susp} 句")
    if word_cnt:
        print("触发词：", ", ".join(f"{w}×{c}" for w, c in word_cnt.most_common(10)))
    if susp_cnt:
        print("可疑结构：", ", ".join(f"{s}×{c}" for s, c in susp_cnt.most_common()))
    raise SystemExit(1 if n_hit else 0)


if __name__ == "__main__":
    main()
