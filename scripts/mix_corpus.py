#!/usr/bin/env python3
"""把几份 (wav/ + transcripts.tsv) 语料合成一份混合语料，供 prepare_data.py 直接吃。

为什么需要：0.92h 教师语料全参数微调在 held-out 上是纯损伤、零收益（见
docs/icefall-finetune-flow.md §7.3）。要挡灾难性遗忘，最便宜的做法是把**基座自己的正确
输出**混进训练集（`gen_teacher_corpus.py --backend v1` 本地几分钟、零云成本，
且只收过生产门的＝V1 读对的那些）。混合是数据侧动作，不占 GPU。

id 冲突处理：各源共用 `u0001…` 编号，直接并会互相覆盖 → 每个源按顺序加 `a_ / b_ / c_`
前缀，`transcripts.tsv` 第三列改写成 `源目录名`，这样产物还能追溯每条来自哪份语料。
wav 直接复制（不建硬链接）：混合语料要 tar+scp 上云，硬链接到归档里会多一类
"解包后端依赖"，而两份语料加起来才 200 多 MB，不值得省。

用法：
  python scripts/mix_corpus.py --out out/corpus_mix out/corpus1k_flat out/replay_v1
"""
import argparse
import os
import shutil
import sys


def read_rows(corpus):
    tsv = os.path.join(corpus, "transcripts.tsv")
    rows = []
    for line in open(tsv, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) >= 2:
            rows.append((p[0].strip(), p[1].strip()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("corpora", nargs="+")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    LETTERS = "abcdefghijklmnopqrstuvwxyz"
    total, dropped = 0, 0
    out_lines = []
    for i, corpus in enumerate(args.corpora):
        rows = read_rows(corpus)
        src = os.path.basename(corpus.rstrip("/"))
        pref = f"{LETTERS[i]}_"
        kept = 0
        for uid, text in rows:
            w = os.path.join(corpus, "wav", uid + ".wav")
            if not os.path.isfile(w):
                dropped += 1
                continue
            new_id = pref + uid
            dst = os.path.join(args.out, "wav", new_id + ".wav")
            if os.path.exists(dst):
                print(f"!! id 撞车：{new_id} 已存在（前缀不够用？）", file=sys.stderr)
                sys.exit(1)
            shutil.copy2(w, dst)
            out_lines.append(f"{new_id}\t{text}\t{src}")
            kept += 1
        print(f"{corpus}: 收 {kept}/{len(rows)} 条（前缀 {pref}）")
        total += kept

    with open(os.path.join(args.out, "transcripts.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"合计 {total} 条 -> {args.out}（缺 wav 被跳过 {dropped} 条）")


if __name__ == "__main__":
    main()
