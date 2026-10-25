#!/usr/bin/env python3
"""为"重复数字串"这一真病灶做定向话术 + 一把**数字值互斥**的验收尺子。

为什么盯数字串：held-out 形态分解（ruler.py 的「形态」列，2026-10-25 实测）显示 V1 的
`时间` 类 30% 错里 **27 个偏差音节全是"复制"、零吞并零替换**——`早晨六点四十分` 被念成
"六点**四**四十分"。这是 token→帧对齐的病，微调能治；`大夫` 那类是前端读音错
（check_frontend.py），微调治不了，不进本文件的靶子。

为什么必须"值互斥"而不是"整句不相交"：实测旧语料把 1..99 的床号读法只漏了
`三十三`/`六十九`，分钟位/小数位/小时位**零空白**；旧尺子 25 条中文行里有 11 条与语料
共享 ≥4 字的数字窗口（`点四十分`、`七十八床`）——那些条上"背题"和"泛化"分不开。
所以本脚本按三条规则造数据：
  1. 先定**尺子独占**的数字取值（HOLD_VALUES，写死、可核对）；
  2. 由尺子句子反推**禁用数字窗口**，训练侧一条都不许碰；
  3. 旧语料里含数字的句子必须整批剔出训练集（顺手产出 data/corpus1k_nodigit.tsv，
     只保留 213 条无数字句做广度），否则前两条白做。

用法：
  python scripts/make_digit_script.py --n 600 --hold-n 16
产物：
  data/record_script_digits.tsv     喂 gen_teacher_corpus.py --script（id<TAB>文本<TAB>类别）
  scripts/ruler_digits.tsv          喂 ruler.py --phrases-file（与主尺子并列跑）
  data/corpus1k_nodigit.tsv         旧语料中不含数字的句（可直接进 mix_corpus）
"""
import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_disjoint as cd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UNIT = "一二三四五六七八九"
MINUTES = ["整", "零五分", "十分", "十五分", "二十分", "二十五分", "三十分",
           "三十五分", "四十分", "四十五分", "五十分", "五十五分"]
PERIOD = ["凌晨", "清晨", "上午", "中午", "下午", "晚间"]
DAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
HOURS = list(UNIT) + ["十", "十一", "十二"]
ACT = ["复查", "换药", "抽血", "雾化", "口服药", "心电监护", "留取标本", "伤口冲洗",
       "床头交班", "送检验科", "拍片", "做彩超"]
BED_SUF = ["呼叫护士站", "请到护士站", "准备接收", "家属请稍等", "输液已开启",
           "需要换纱布", "请调整体位", "叫号已确认"]
NAME = ["赵国强", "孙丽华", "周秀英", "吴桂芳", "郑玉梅", "冯志强", "陈建国",
        "褚小红", "卫淑珍", "蒋凤英", "沈德福", "韩春梅", "杨国庆", "朱小明"]
PROC = ["输液", "雾化", "抽血", "换药", "拍片", "彩超", "口服药", "心电监护"]
DOSE_UNIT = ["毫升", "毫克", "片", "支"]

# ---- 尺子独占的数字取值（三条轴各留一块，全部落在旧语料的空白位上） ----
HOLD_VALUES = {
    "min": ["三十分", "五十分"],                 # 语料 0 例
    "dup": ["三十三", "六十九"],                 # 1..99 床号里语料仅有的两个空白读法
    "bed3": ["一百零三", "一百三十八", "一百六十九", "一百一十", "一百八十"],
    "dec": ["三", "七"],                         # 小数位；旧语料有，所以旧数字句要整批剔除
}


def bed_spoken(b):
    """1..199 → 播报读法（零五床 / 三十一床 / 一百零三床 / 一百二十床）。"""
    if b < 10:
        return "零" + UNIT[b - 1]
    if b == 10:
        return "十"
    if b < 20:
        return "十" + UNIT[b - 11]
    if b < 100:
        t, u = divmod(b, 10)
        return UNIT[t - 1] + "十" + (UNIT[u - 1] if u else "")
    h, r = divmod(b, 100)
    if r == 0:
        return UNIT[h - 1] + "百"
    if r < 10:
        return UNIT[h - 1] + "百零" + UNIT[r - 1]
    if r < 20:
        return UNIT[h - 1] + "百一十" + (UNIT[r - 11] if r > 11 else "")
    return UNIT[h - 1] + "百" + bed_spoken(r)


def one_time(rng, mins, with_name=True):
    m, h, p = rng.choice(mins), rng.choice(HOURS), rng.choice(PERIOD)
    if with_name and rng.random() < 0.45:
        return (f"{rng.choice(NAME)}的{rng.choice(PROC)}安排在{rng.choice(DAYS)}{p}{h}点{m}",
                "时间数字")
    return (f"{p}{h}点{m}{rng.choice(ACT)}", "时间数字")


def one_bed(rng, beds, with_name=True):
    b = rng.choice(beds)
    if not with_name or rng.random() < 0.5:
        return (f"{b}床{rng.choice(BED_SUF)}", "床号数字串")
    return (f"{b}床患者{rng.choice(NAME)}{rng.choice(BED_SUF)}", "床号数字串")


def one_dose(rng, decs, with_name=True):
    tail = f"，{rng.choice(NAME)}服用后请漱口" if with_name else "，服用后请漱口"
    return (f"每次{rng.choice(UNIT)}点{rng.choice(decs)}{rng.choice(DOSE_UNIT)}"
            f"{tail}", "剂量小数")


def fill(rng, quota, slots, forbid=None, with_name=True):
    """按类别配额造句：去重、避开 forbid 里的数字窗口；某类填不满就少给并报出来。"""
    gen = {"min": lambda: one_time(rng, slots["min"], with_name),
           "bed": lambda: one_bed(rng, slots["bed"], with_name),
           "dec": lambda: one_dose(rng, slots["dec"], with_name)}
    out, short = [], {}
    for kind, want in quota.items():
        if not slots.get(kind):
            short[kind] = want
            continue
        got, seen, tries = [], set(), 0
        while len(got) < want and tries < want * 400:
            tries += 1
            t, c = gen[kind]()
            if t in seen or (forbid and clash(t, forbid)):
                continue
            seen.add(t)
            got.append((t, c))
        out += got
        if len(got) < want:
            short[kind] = want - len(got)
    rng.shuffle(out)
    return out, short


def gram_set(texts, n=4):
    g, p = set(), set()
    for t in texts:
        g |= cd.dgrams(t, n)
        p |= cd.pdgrams(t, n)
    return g, p


def clash(text, gs):
    for g in cd.dgrams(text):
        if g in gs[0]:
            return "".join(g)
    for g in cd.pdgrams(text):
        if g in gs[1]:
            return "/".join(g)
    return None


def write_lines(path, rows, fmt, note):
    with open(path, "w", encoding="utf-8") as f:
        f.write(note + "\n")
        for i, r in enumerate(rows, 1):
            f.write(fmt(i, r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="训练侧条数")
    ap.add_argument("--hold-n", type=int, default=16, help="尺子侧条数")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--corpus", default=os.path.join(REPO, "out", "corpus1k", "transcripts.tsv"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "record_script_digits.tsv"))
    ap.add_argument("--holdout-out", default=os.path.join(REPO, "scripts", "ruler_digits.tsv"))
    ap.add_argument("--nodigit-out", default=os.path.join(REPO, "data", "corpus1k_nodigit.tsv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hv = HOLD_VALUES
    hold_quota = {"min": max(1, args.hold_n // 2), "bed": max(1, args.hold_n * 3 // 10),
                  "dec": args.hold_n - max(1, args.hold_n // 2) - max(1, args.hold_n * 3 // 10)}
    # 1) 尺子侧：只用独占取值，不需要过滤（它就是禁用窗口的来源）
    # 尺子侧一律**不带人名**：实测带人名的句子把读数污染成"人名病"
    # （`六十九床患者孙丽华…` 72.5% 错、错处全在 孙→斯；无人名的 `上午一点三十分拍片` 0%）
    held, hold_short = fill(rng, hold_quota,
                            {"min": hv["min"], "bed": hv["dup"] + hv["bed3"], "dec": hv["dec"]},
                            with_name=False)
    held.sort(key=lambda r: r[1])
    by_axis = collections.Counter(c for _, c in held)
    forbid = gram_set([t for t, _ in held])

    # 2) 训练侧：全集减去尺子独占取值，且避开尺子的数字窗口。
    #    注意**不对旧语料的数字句做过滤**——那 762 句本来就要被剔出训练集，
    #    拿它当禁用集会把训练侧压成 419/600（实测），那是自己给自己上枷锁。
    train_slots = {"min": [m for m in MINUTES if m not in hv["min"]],
                   "bed": [bed_spoken(b) for b in range(1, 200)
                           if bed_spoken(b) not in set(hv["dup"]) | set(hv["bed3"])],
                   "dec": [u for u in UNIT if u not in hv["dec"]]}
    train_quota = {"min": args.n * 4 // 10, "bed": args.n * 4 // 10, "dec": args.n // 5}
    train, train_short = fill(rng, train_quota, train_slots, forbid, with_name=True)

    old = cd.read_texts(args.corpus, 2) if os.path.exists(args.corpus) else []
    old_digit = [t for _, t in old if cd.DIGITS & set(cd.norm(t))]
    old_free = [(i, t) for i, t in old if not (cd.DIGITS & set(cd.norm(t)))]

    write_lines(args.holdout_out, held, lambda i, r: f"{r[0]}\t{r[0]}\t{r[1]}",
                "# 定向数字尺子：取值与训练侧互斥（make_digit_script.py 生成并自检）\n"
                "# ⚠ 训练集里不得出现旧语料的含数字句（见 data/corpus1k_nodigit.tsv）\n"
                "# 格式：喂入<TAB>期望<TAB>类别")
    write_lines(args.out, train, lambda i, r: f"dg{i:04d}\t{r[0]}\t{r[1]}",
                f"# 重复数字串定向话术（seed={args.seed}）；列: id<TAB>文本<TAB>类别")
    write_lines(args.nodigit_out, old_free, lambda i, r: f"{r[0]}\t{r[1]}",
                "# 旧语料里不含数字的句子（可直接进 mix_corpus；含数字的 "
                f"{len(old_digit)} 句必须剔除，否则数字值留出作废）")

    print(f"[digit] 尺子独占取值: {hv}")
    print(f"[digit] 尺子 {len(held)} 条 → {os.path.relpath(args.holdout_out, REPO)}  "
          f"轴分布 {dict(by_axis)}")
    for t, c in held:
        print(f"        {c:<8} {t}")
    print(f"[digit] 训练 {len(train)} 条 → {os.path.relpath(args.out, REPO)}  "
          f"类别 {dict(collections.Counter(c for _, c in train))}")
    print(f"[digit] 旧语料 {len(old)} 句拆成：含数字 {len(old_digit)}（剔除出训练集）、"
          f"无数字 {len(old_free)}（保留，清单 {os.path.relpath(args.nodigit_out, REPO)}）")
    if train_short:
        print(f"!! 训练侧缺口 {train_short}——尺子占掉的数字空间太大，调小 --hold-n")
    if hold_short:
        print(f"!! 尺子缺口 {hold_short}——独占取值不够，扩 HOLD_VALUES")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
