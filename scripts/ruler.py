#!/usr/bin/env python3
"""尺子：每条话术重复合成 N 次，SenseVoice 回读，统计内容错率与错误分布。

模型是扩散采样、不稳定，单次回读不算结论。音频只在内存过 ASR，默认不落盘。
喂入用规整话术（字母拆 I;C;U;、数字转汉字）。字母行注意：ASR 会把字母名拼成
单词(i c you)，属度量假错，看中文行即可。
两个口径同时报：**字**＝严格字符比对（含 ASR 同音选字假错，如 输液→书叶）；
**音**＝拼音口径（asr.py 的 py），只算读音真错。两者差多少＝假错有多少；
对比两个模型时，涨幅只出现在字口径多半是度量噪声，音口径一起涨才是真损伤。
"""
import argparse
import collections
import os

import asr
import config
import repro_v1 as rp

DEFAULT = [
    ("体温三十六点五度", "体温三十六点五度"),
    ("为您自动接听", "为您自动接听"),
    ("二十一床正在呼叫", "二十一床正在呼叫"),
    ("A;一零床正在呼叫", "A一零床正在呼叫"),
    ("I;C;U;三床请到护士站", "ICU三床请到护士站"),
]


def load_phrases(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        feed, _, exp = line.partition("\t")
        out.append((feed.strip(), (exp or feed).strip()))
    return out


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
    tot_err = tot_err_py = tot_n = 0
    for feed, exp in phrases:
        expn = asr.norm(exp)
        expp = asr.py(exp)
        cnt = collections.Counter()
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
            if asr.py(heard) != expp:
                err_py += 1
        tot_err += err
        tot_err_py += err_py
        tot_n += args.n
        print(f"[{err}/{args.n} 字={err/args.n*100:5.1f}% 音={err_py/args.n*100:5.1f}%] {feed!r}")
        for heard, c in cnt.most_common(4):
            tag = "OK " if asr.norm(heard) == expn else ("同音" if asr.py(heard) == expp else "ERR")
            print(f"    {tag} x{c:<3d} {heard}")
    print(f"\nTOTAL n={tot_n}  字口径 err={tot_err} = {tot_err/tot_n*100:.1f}%"
          f"   音口径 err={tot_err_py} = {tot_err_py/tot_n*100:.1f}%")


if __name__ == "__main__":
    main()
