#!/usr/bin/env python3
"""候选人发音筛选：读 (wav + transcripts.tsv)，SenseVoice 回读，量"念得清不清楚"。

正式录 1-2 小时前，先让候选人照 data/record_script.tsv 念 20-30 句，跑这个打分。
错率高 = 发音不够标准/含糊，不适合当 TTS 训练声。ASR 自身也会错，这里看"相对清晰度"，
最终仍要人耳抽听。
"""
import argparse
import os
from math import gcd

import soundfile as sf

import asr


def load_tsv(corpus):
    texts = {}
    for line in open(os.path.join(corpus, "transcripts.tsv"), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) >= 2:
            texts[p[0].strip()] = p[1].strip()
    return texts


def to16k(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a[:, 0]
    if sr != 16000:
        try:
            from scipy.signal import resample_poly
            g = gcd(16000, sr)
            a = resample_poly(a, 16000 // g, sr // g).astype("float32")
        except Exception:
            print(f"[warn] {os.path.basename(path)} sr={sr}, 未重采样")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="含 wav/ 与 transcripts.tsv")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    rec = asr.load_recognizer(args.threads)
    texts = load_tsv(args.corpus)
    wav_dir = os.path.join(args.corpus, "wav")
    n = err = 0
    bad = []
    for uid in sorted(texts):
        wp = os.path.join(wav_dir, uid + ".wav")
        if not os.path.isfile(wp):
            print(f"[skip] no wav {uid}")
            continue
        heard = asr.transcribe(rec, to16k(wp))
        n += 1
        if asr.norm(heard) != asr.norm(texts[uid]):
            err += 1
            bad.append((uid, texts[uid], heard))
    print(f"\n样本={n}  不匹配={err}  错率={err/max(n,1)*100:.1f}%")
    print("（先 <10% 再录全量；最终人耳抽听）")
    for uid, exp, heard in bad:
        print(f"  {uid}: 期望[{exp}] 听到[{heard}]")


if __name__ == "__main__":
    main()
