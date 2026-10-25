#!/usr/bin/env python3
"""尺子：每条话术重复合成 N 次，SenseVoice 回读，统计内容错率与错误分布。

模型是扩散采样、不稳定，单次回读不算结论。音频只在内存过 ASR，默认不落盘。
喂入用规整话术（字母拆 I;C;U;、数字转汉字）。字母行注意：ASR 会把字母名拼成
单词(i c you)，属度量假错，看中文行即可。
两个口径同时报：**字**＝严格字符比对（含 ASR 同音选字假错，如 输液→书叶）；
**音**＝拼音口径（asr.py 的 py），只算读音真错。两者差多少＝假错有多少；
对比两个模型时，涨幅只出现在字口径多半是度量噪声，音口径一起动才是真变化。

第三个维度是**错误形态**（按音节做序列对齐）：`复制`=多念出相邻已有的音节、
`吞并`=少音节、`替换`=音节数不变但换了。首轮微调的损伤与前 5pp 的真痛点都落在
复制/吞并上（`六点四十分`→"六点四四十分"），单看总错率会把"读错音"和"对齐塌了"
混成一个数——它们要的修法完全不同（前者是数据覆盖，后者是时长头）。
另外，前端本身读错的词（check_frontend.py 的口径）会打 `[前端]` 标记：
这类条目的错**不能归因给声学模型**，微调治不了它。
"""
import argparse
import collections
import difflib
import os

import asr
import check_frontend as cfr
import config
import repro_v1 as rp

DEFAULT = [
    ("体温三十六点五度", "体温三十六点五度", "未分类"),
    ("为您自动接听", "为您自动接听", "未分类"),
    ("二十一床正在呼叫", "二十一床正在呼叫", "未分类"),
    ("A;一零床正在呼叫", "A一零床正在呼叫", "字母/英文"),
    ("I;C;U;三床请到护士站", "ICU三床请到护士站", "字母/英文"),
]


def load_phrases(path):
    """三列：喂入 <TAB> 期望(缺省同喂入) <TAB> 类别(缺省「未分类」)。"""
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        c = line.split("\t")
        feed = c[0].strip()
        exp = (c[1].strip() if len(c) > 1 and c[1].strip() else feed)
        tag = c[2].strip() if len(c) > 2 and c[2].strip() else "未分类"
        out.append((feed, exp, tag))
    return out


def error_modes(exp, heard):
    """一次回读相对期望的音节级偏差 → (复制, 插音, 吞并, 替换) 四个计数。

    `复制`=插入的音节与期望中相邻位置的某个音节相同（"六点四**四**十分"），
    `插音`=插入了邻居里没有的音节，`吞并`=少了音节，`替换`=音节数不变但换了。
    首轮微调与前 5pp 真痛点的形态是复制/吞并，不是替换。
    """
    a, b = list(exp), list(heard)
    dup = ins = drop = subs = 0
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op == "insert":
            near = set(a[max(0, i1 - 1):i2 + 1])
            for x in b[j1:j2]:
                dup += 1 if x in near else 0
                ins += 0 if x in near else 1
        elif op == "delete":
            drop += i2 - i1
        else:
            la, lb = i2 - i1, j2 - j1
            subs += min(la, lb)
            ins += max(0, lb - la)
            drop += max(0, la - lb)
    return dup, ins, drop, subs



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--phrases-file", default="")
    ap.add_argument("--sample-out", default="")
    args = ap.parse_args()

    rec = asr.load_recognizer(args.threads)
    am, voc, tok2id = rp.build(args.threads)
    if args.sample_out:
        os.makedirs(args.sample_out, exist_ok=True)

    phrases = load_phrases(args.phrases_file) if args.phrases_file else DEFAULT
    ovr = cfr.load_overrides()
    tot_err = tot_err_py = tot_n = 0
    mode_tot = collections.Counter()
    cat = collections.defaultdict(lambda: [0, 0])       # 类别 -> [样本数, 音口径错数]
    fe_bad = []
    for feed, exp, tag in phrases:
        expn = asr.norm(exp)
        expp = asr.py(exp)
        cnt = collections.Counter()
        modes = collections.Counter()
        err = err_py = 0
        for i in range(args.n):
            _, audio = rp.synth(am, voc, tok2id, feed)
            if i == 0 and args.sample_out:
                import soundfile as sf
                sf.write(os.path.join(args.sample_out, rp.safe(feed) + ".wav"),
                         audio, am.sample_rate, "PCM_16")
            heard = asr.transcribe(rec, audio, am.sample_rate)
            cnt[heard] += 1
            if asr.norm(heard) != expn:
                err += 1
            hp = asr.py(heard)
            if hp != expp:
                err_py += 1
                for k, v in zip(("复制", "插音", "吞并", "替换"), error_modes(expp, hp)):
                    modes[k] += v
        tot_err += err
        tot_err_py += err_py
        tot_n += args.n
        mode_tot.update(modes)
        cat[tag][0] += args.n
        cat[tag][1] += err_py
        if cfr.audit(exp, ovr)[0]:
            fe_bad.append(exp)
        fe = "  [前端读错，别归因给模型]" if exp in fe_bad else ""
        shape = " ".join(f"{k}{v}" for k, v in modes.most_common()) or "—"
        print(f"[{err}/{args.n} 字={err/args.n*100:5.1f}% 音={err_py/args.n*100:5.1f}%"
              f" {tag} 形态:{shape}] {feed!r}{fe}")
        for heard, c in cnt.most_common(4):
            tag2 = "OK " if asr.norm(heard) == expn else ("同音" if asr.py(heard) == expp else "ERR")
            print(f"    {tag2} x{c:<3d} {heard}")
    print(f"\nTOTAL n={tot_n}  字口径 err={tot_err} = {tot_err/tot_n*100:.1f}%"
          f"   音口径 err={tot_err_py} = {tot_err_py/tot_n*100:.1f}%")
    print("形态合计(按音节):", " ".join(f"{k}={mode_tot[k]}"
                                      for k in ("复制", "插音", "吞并", "替换")))
    print("分类别音口径:", "  ".join(f"{k}={v[1]/v[0]*100:.0f}%(n={v[0]})"
                                  for k, v in sorted(cat.items(), key=lambda kv: -kv[1][1]/kv[1][0])))
    if fe_bad:
        print(f"⚠ 其中 {len(fe_bad)} 条前端读音本身是错的（微调治不了）：", "、".join(fe_bad))


if __name__ == "__main__":
    main()
