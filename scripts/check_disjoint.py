#!/usr/bin/env python3
"""尺子纯净度检查：验收用的短语绝不能以任何形式出现在训练语料里。

首轮就栽过一次：尺子和语料同取自一份生产短语表，`体温三十六点五度` 在 train 切分里
逐字出现 3 次，微调后该条 45%→0% 看着像"治好了"，其实是背题。所以判"提升"之前，
这四条都必须为 0：
  整句     尺子句 == 语料句
  子串     尺子句里有 ≥--ngram 个连续字与语料某句相同（半背题）
  拼音整句  读音完全同、字不同（换了写法还是同一句）
  拼音子串  同理，但只看 ≥--ngram 的连续音节

用法：
  python scripts/check_disjoint.py --ruler scripts/ruler_heldout.tsv \
      --corpus out/corpus1k/transcripts.tsv
退出码 0 = 干净，可以拿这把尺子判收益。
"""
import argparse
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pypinyin import lazy_pinyin, Style

_STRIP = re.compile(r"[\s，。、；;,.!！?？:：'\"“”()（）\-]")


def norm(s):
    return _STRIP.sub("", s.lower())


def syl(s):
    return tuple(lazy_pinyin(norm(s), style=Style.TONE3,
                             neutral_tone_with_five=True, errors="default"))


def grams(seq, n):
    """连续 n 元组集合；seq 可以是字符串或音节列表（元素统一成可哈希的 tuple）。"""
    if len(seq) < n:
        return set()
    return {tuple(seq[i:i + n]) if not isinstance(seq, str) else seq[i:i + n]
            for i in range(len(seq) - n + 1)}


DIGITS = set("零一二三四五六七八九十百千万幺两0123456789")   # 阿拉伯数字也算，别让它漏网
# weekday / 初几 里的数词不是"数字串"，别让它参与数字窗口判定：`排在周五` 撞 `周三`
# 只是共用一个星期词尾，与复制/吞并无关（实测误伤过 1381 条训练候选里的一大片）。
_NONNUM = re.compile(r"(?:周|星期|初)[一二三四五六七八九十日天]")


def nummask(s):
    return _NONNUM.sub(lambda m: m.group(0)[:-1] + "＊", s)


def dgrams(text, n=4):
    """只取**含数字**的 n 字窗口。

    子串污染要区分两种重合：话术模板的重合（`呼叫护士站` 语料里几百次，但它不是考点）
    和数字串的重合（`点四十分`——这才是半背题）。用整句口径会把前者也算成污染
    （28 条旧尺子 12 条"中招"，其中确实有真数字重合，也有纯模板重合），
    新尺子根本造不出来；用数字窗口口径只拦后者。
    """
    t = nummask(norm(text))
    return {g for g in grams(t, n) if DIGITS & set(g)}


def pdgrams(text, n=4):
    """数字窗口对应的**音节** n 元组——抓换字不换音的半背题（`四十分`↔`拾分`）。

    一字一音节不成立（句中含字母/英文）时返回空集，即该条只做整句级拼音检查。
    """
    t = nummask(norm(text))
    s = syl(t)
    if len(s) != len(t):
        return set()
    return {tuple(s[i:i + n]) for i in range(len(t) - n + 1)
            if DIGITS & set(t[i:i + n])}


def read_texts(path, col):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) >= col:
            out.append((c[0], c[col - 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler", default="scripts/ruler_heldout.tsv")
    ap.add_argument("--corpus", required=True, action="append",
                    help="语料 transcripts.tsv，可重复给多份（尺子要对全部训练数据求交集）")
    ap.add_argument("--ruler-col", type=int, default=2)
    ap.add_argument("--corpus-col", type=int, default=2)
    ap.add_argument("--ngram", type=int, default=4)
    ap.add_argument("--whole-window", action="store_true",
                    help="子串口径不过滤数字（连模板重合也算污染，一般太严，新尺子会造不出来）")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    ruler = read_texts(args.ruler, args.ruler_col)
    c_texts, c_syls, c_cg, c_pg = [], [], set(), set()
    n = args.ngram
    nf = 0
    for path in args.corpus:
        for _, t in read_texts(path, args.corpus_col):
            t2 = norm(t)
            c_texts.append(t2)
            c_syls.append("\t".join(syl(t2)))
            c_cg |= (grams(t2, n) if args.whole_window else dgrams(t2, n))
            c_pg |= set() if args.whole_window else pdgrams(t2, n)
        nf += 1

    k_sub = "子串" if args.whole_window else "数字子串"
    k_psub = "拼音子串" if args.whole_window else "拼音数字子串"
    bad = {"整句": [], k_sub: [], "拼音整句": [], k_psub: []}
    for rid, rtext in ruler:
        t = norm(rtext)
        s = syl(t)
        if t in c_texts:
            bad["整句"].append((rid, rtext))
        if "\t".join(s) in c_syls:
            bad["拼音整句"].append((rid, rtext))
        wg = grams(t, n) if args.whole_window else dgrams(rtext, n)
        hit = [g for g in wg if g in c_cg]
        if hit:
            bad[k_sub].append((rid, rtext, "".join(hit[0])))
        hitp = [g for g in (set() if args.whole_window else pdgrams(rtext, n)) if g in c_pg]
        if hitp:
            bad[k_psub].append((rid, rtext, "/".join(hitp[0])))

    total = 0
    for k, v in bad.items():
        total += len(v)
        print(f"{k:<10} 命中 {len(v):>3} 条", end="")
        if v:
            print("  例: " + " | ".join(f"{x[1]}⟨{x[2] if len(x) > 2 else ''}⟩"
                                        for x in v[:args.show]))
        else:
            print()
    print(f"\n尺子 {len(ruler)} 条 × 语料 {len(c_texts)} 条（{nf} 份），n-gram={n}，"
          f"窗口口径={'全部' if args.whole_window else '仅含数字'}：" +
          ("干净，可用于判收益" if total == 0 else f"❌ {total} 处污染，判读作废"))
    raise SystemExit(1 if total else 0)


if __name__ == "__main__":
    main()
