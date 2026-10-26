#!/usr/bin/env python3
"""把一把尺子展开成多种"表面形式"，用来测：不改模型、只改喂进去的写法，错率降不降。

动机（实测出来的）：V1 在定向数字尺上错 24.1%，形态是 复制12/插音14/吞并65/替换109，
错集中在**相邻数字音节被合并或重复**（`一百三十八床`→"一百三床"、`六点四十`→"六四四十"）。
微调两轮判死、换基座 RTF 判死，所以还剩一条不花钱的路：既然病灶是"音节串太长太像"，
那就在**写法上把它拆开**——加停顿、缩短、换读法。本脚本只负责生成变体尺子，
打分一律走 scripts/ruler.py（与所有历史基线同一套口径，别自己另写评分）。

变体与假设：
  pause   数字串后插一个逗号 —— 假设：停顿给 duration 头一个边界，压制合并/重复
  split   原逗号改句号       —— 假设：缩短注意力窗口（句内长依赖是病灶）
  digit   床号改逐位读       —— 一百三十八床→一三八床：减少音节数与相邻重复（**内容变了**，
          所以期望文本也跟着变；这是产品可选项，不是同一句话的等价改写）
  orig    原句（对照，用来确认脚本没把分数算错：应与登记基线 24.1% 同量级）

用法：
  python scripts/surface_forms.py --in scripts/ruler_digits.tsv --out-dir out/surface
  然后对每个变体跑 ruler.py（见打印出来的命令）。
"""
import argparse
import os
import re

CN = "零一二三四五六七八九十百千两幺"
DIGIT_OF = {"零": "0", "一": "1", "幺": "1", "二": "2", "两": "2", "三": "3", "四": "4",
            "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
# 逐位读回写用的汉字表：固定用"一二"，不用 幺/两（那两种是另一种读法，别混进这个变体）
UNIT_OF = {str(i): c for i, c in enumerate("零一二三四五六七八九")}
NUM = re.compile(f"[{CN}点]+")
# 停顿该落在"整个数量短语"之后，而不是把数量短语切开。所以单位词要列全：
# 漏一个"度"就会生成出 `体温三十七点二。度` 这种句子（实测就这么造出过一个假退化）。
UNITS = sorted((
    "摄氏度", "华氏度", "毫克", "毫升", "公斤", "千克", "厘米", "毫米", "分钟", "分秒", "小时",
    "次", "床", "片", "粒", "支", "袋", "岁", "号", "度", "克", "升", "米", "天", "人", "例", "倍", "元",
    "分", "秒",
), key=len, reverse=True)


def de_punct(s):
    return re.sub(r"[，。！？、；]", "", s)


def num_span(text):
    """返回 (数字串, 含单位的短语结束位置)。解析不出数字时返回 (None, None)。"""
    m = NUM.search(text)
    if not m:
        return None, None
    end = m.end()
    for u in UNITS:
        if text[end:end + len(u)] == u:
            end += len(u)
            break
    return m.group(0), end


def parse_cn(s):
    """支持 ≤999 的基数读法（够床号用）；解析不了返回 None，绝不猜。"""
    if not s or any(c not in CN for c in s):
        return None
    total = 0
    if "百" in s:
        head, s = s.split("百", 1)
        if not head:
            return None
        n = DIGIT_OF.get(head)
        if n is None:
            return None
        total += int(n) * 100
        s = s.lstrip("零")
    if not s:
        return total
    if "十" in s:
        head, s = s.split("十", 1)
        if head:
            n = DIGIT_OF.get(head)
            if n is None:
                return None
            total += int(n) * 10
        else:
            total += 10
        s = s.lstrip("零")
    if s:
        if len(s) != 1:
            return None
        n = DIGIT_OF.get(s)
        if n is None:
            return None
        total += int(n)
    return total if total > 0 else None


def render_digits(n):
    return "".join(UNIT_OF[d] for d in str(n))


def variants(text, tag):
    """返回 {变体名: (喂入, 期望)}；期望默认=原期望，只有 digit 会改期望。"""
    m = NUM.search(text)
    run, end = num_span(text)
    # 尾巴不足 2 字就别在中间下刀（否则又会切出 `…点二。度` 这类单字尾巴）
    cut = end if (run is not None and len(text) - end >= 2) else None
    out = {}
    # 已经有标点的地方不再叠第二层（"每次六点七毫克，" 本来就有停顿，那这句的 pause 变体=原句）
    if cut and text[cut] not in "，。！？、；":
        out["pause"] = (text[:cut] + "，" + text[cut:], text)
    else:
        out["pause"] = (text, text)
    if "，" in text:
        out["split"] = (text.replace("，", "。", 1), text)
    elif cut:
        out["split"] = (text[:cut] + "。" + text[cut:], text)
    else:
        out["split"] = (text, text)
    if tag.startswith("床号") and m:
        n = parse_cn(m.group(0))
        if n is None:
            out["digit"] = (text, text)          # 解析不了就退回原句，不硬编
        else:
            new = text[:m.start()] + render_digits(n) + text[m.end():]
            out["digit"] = (new, new)            # 期望跟着变：我们要的就是逐位读被念对
    # 下面两个是**对照**：用来分清"边界位置"和"任意停顿"，以及"一个边界"和"全切短"
    if run:
        out["pause_before"] = (text[:m.start()] + "，" + text[m.start():], text)
    else:
        out["pause_before"] = (text, text)
    # 逗号→句号是 1:1 替换，长度不变，所以 cut 这个下标在 pa 里仍然对齐
    pa = "".join("。" if c == "，" else c for c in text)
    if cut and pa[cut] not in "，。！？、；":
        pa = pa[:cut] + "。" + pa[cut:]
    out["period_all"] = (pa, text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="scripts/ruler_digits.tsv")
    ap.add_argument("--out-dir", default="out/surface")
    args = ap.parse_args()

    rows = []
    for line in open(args.infile, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.rstrip("\n").split("\t")
        rows.append((p[0], p[1] if len(p) > 1 else p[0], p[2] if len(p) > 2 else "未分类"))
    os.makedirs(args.out_dir, exist_ok=True)
    seen = {}
    for feed, exp, tag in rows:
        d = dict(variants(feed, tag))
        d["orig"] = (feed, exp)
        for k, (f, e) in d.items():
            seen.setdefault(k, []).append((f, e, tag, feed))
    for k, items in seen.items():
        path = os.path.join(args.out_dir, f"{k}.tsv")
        # 自检（digit 除外，它是合法地换了读法）：只许插标点，不许增删任何一个字。
        # 这条就是用来拦我自己犯过的错：单位词表漏一个"度"，就会生成出 `三十七点二。度`。
        # 自检（只针对**本脚本生成的**变体；orig 是原始行、digit 是合法换读法，都不该套这条）：
        # 只许插标点，不许增删任何一个字。这条就是用来拦我自己犯过的错：
        # 单位词表漏一个"度"，就会生成出 `三十七点二。度`。
        if k not in ("digit", "orig"):
            for feed, exp, tag, orig in items:
                assert de_punct(feed) == de_punct(exp), f"{k} 变体动了字：{feed!r} vs {exp!r}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 由 surface_forms.py 从 {args.infile} 生成的 {k} 变体\n")
            for feed, exp, tag, _ in items:
                f.write(f"{feed}\t{exp}\t{tag}\n")
        nch = sum(1 for feed, _, _, orig in items if feed != orig)
        print(f"[{k:5s}] {len(items):3d} 句（其中真正改了写法 {nch} 句）→ {path}")
    print("\n打分（与所有历史基线同一口径，n 与基线一致才可比）：")
    for k in seen:
        print(f"  .venv/bin/python scripts/ruler.py --n 20 --threads 4 "
              f"--phrases-file {args.out_dir}/{k}.tsv | tail -4")


if __name__ == "__main__":
    main()
