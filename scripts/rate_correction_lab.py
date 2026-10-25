#!/usr/bin/env python3
"""语料语速校正的候选算法选型：同一条句子用不同办法拉慢，比**内容是否还过得去**和
**频谱离原声多远**，再决定用哪种。

背景：`flatten_rate.py` 用 librosa 相位声码器（默认 n_fft=2048，对 16k 语音窗太粗）
把 >13 音节的长句拉慢 12~15%，结果生产时那个 ASR 门的通过率从 79/80 掉到 59/80，
掉的是音节（`二十床的吸氧安排`→"二十床的线安排"）——即伪影直接改变了内容。
把窗调细（n_fft=512）能回到 75/80，但仍有损耗。这个脚本回答两件事：
① 时域 SOLA/WSOLA 是不是比相位声码器更适合（它不动音高、只重定位移）；
② 如果都过不了门，就只剩"老师侧 --speed 重合成"这一条路（要 GPU 实例上的权重）。

用法：
  python scripts/rate_correction_lab.py --corpus out/corpus1k --n 40
"""
import argparse
import collections
import os
import sys

import numpy as np
import soundfile as sf
from pypinyin import lazy_pinyin, Style

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr
from gen_teacher_corpus import _HAS_LETTER, pinyin_gate

TARGET = 0.283          # 秒/音节：语料 5~10 音节档实测，且与 V1 自身 0.284 对齐


def syllables(text):
    return lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True, errors="default")


def load(corpus):
    texts = {}
    for line in open(os.path.join(corpus, "transcripts.tsv"), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) >= 2:
            texts[p[0].strip()] = p[1].strip()
    return texts


def stretch_pv(y, sr, factor, n_fft=512, hop=None):
    """相位声码器：时频重映射，保音高，长窗会糊掉辅音瞬态。"""
    import librosa
    hop = hop or n_fft // 4
    return librosa.effects.time_stretch(y, rate=1.0 / factor, n_fft=n_fft, hop_length=hop)


def stretch_resample(y, sr, factor):
    """变调变速（磁带式拉慢）：只做带限插值把波形拉长，**零相位伪影**，代价是音高与
    共振峰整体下移。列进来是为了量化"变调"这条代价有多大——单音色模型要保住 V1 的音色，
    所以它能不能用取决于频谱距离与 ASR 通过率，不预设结论。"""
    from scipy.signal import resample_poly
    m, n = 1000, int(round(1000 / factor))
    return resample_poly(y, m, n).astype(np.float32)


def stretch_sola(y, sr, factor, win_ms=30, max_shift_ms=8):
    """时域 SOLA：帧步进按 1/factor 压缩，每帧在 ±max_shift 内找与"上一帧已写出的尾巴"
    最相关的位置再叠加。输出永远是原波形的片段拼接，不产生相位声码器那种糊掉辅音的失真。"""
    y = y.astype(np.float64)
    win = int(sr * win_ms / 1000)
    hop = win // 2
    prev = win - hop
    step = max(1, int(round(hop / factor)))
    shift = int(sr * max_shift_ms / 1000)
    n = len(y)
    # 输出比输入长 factor 倍；缓冲按 factor 留足，否则 pos_out 会跑过末尾、切片被截短
    out = np.zeros(int(n * max(factor, 1.0)) + win)
    w = np.hanning(win)
    pos_out = pos_in = 0
    while pos_in + win <= n:
        ref = out[pos_out:pos_out + prev]
        rd = ref - ref.mean()
        rn = float(np.sqrt(np.dot(rd, rd)))
        best_off, best_r = 0, -2.0
        for off in range(-shift, shift + 1):
            s = pos_in + off
            if s < 0 or s + win > n:
                continue
            cd = y[s:s + prev] - y[s:s + prev].mean()
            r = float(np.dot(cd, rd)) / (float(np.sqrt(np.dot(cd, cd))) * rn + 1e-9)
            if r > best_r:
                best_r, best_off = r, off
        s = pos_in + best_off
        seg = y[s:s + win]
        L = min(win, len(seg), len(out) - pos_out)
        out[pos_out:pos_out + L] = out[pos_out:pos_out + L] * (1 - w[:L]) + seg[:L] * w[:L]
        pos_out += hop
        pos_in += step
    return out[:pos_out].astype(np.float32)


def spectral_dist(a, b, sr, n_fft=512):
    """归一化幅度谱的平均 L1 距离（越小越像原声）。两条长度不同也能比：
    各自按帧排序后插值到同一帧数，只比谱形不比时长。"""
    def mag(x):
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        stft = np.abs(np.stack([np.fft.rfft(x[i:i + n_fft] * np.hanning(n_fft))
                                for i in range(0, len(x) - n_fft, n_fft // 2)])).T
        m = stft.mean(axis=1)
        return m / (m.sum() + 1e-9)

    def to_grid(v, k=64):
        return np.interp(np.linspace(0, len(v) - 1, k), np.arange(len(v)), v)
    ga, gb = to_grid(mag(a)), to_grid(mag(b))
    return float(np.abs(ga - gb).sum() / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="out/corpus1k")
    ap.add_argument("--n", type=int, default=40, help="抽多少条最赶的长句做对比")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="/tmp/rate_lab")
    args = ap.parse_args()

    texts = load(args.corpus)
    rows = []
    for uid, t in texts.items():
        p = os.path.join(args.corpus, "wav", uid + ".wav")
        if not os.path.isfile(p):
            continue
        y, sr = sf.read(p, dtype="float32")
        syl = syllables(t)
        if len(syl) >= 13:
            rows.append((uid, len(syl), len(y) / sr, len(y) / sr / len(syl)))
    rows.sort(key=lambda r: r[3])
    pick = rows[:args.n]
    print(f"取最赶的 {len(pick)} 条长句（≥13 音节），未校语速 "
          f"{np.mean([r[3] for r in pick]):.3f} 秒/音节 → 目标 {TARGET}")

    methods = collections.defaultdict(lambda: [0, 0, []])
    os.makedirs(args.out, exist_ok=True)
    rec = asr.load_recognizer(args.threads)
    for idx, (uid, nsyl, dur, rate) in enumerate(pick):
        y, sr = sf.read(os.path.join(args.corpus, "wav", uid + ".wav"), dtype="float32")
        need = TARGET / rate
        exp = asr.norm(texts[uid])
        hl = bool(_HAS_LETTER.search(texts[uid]))
        cands = {
            "pv512": stretch_pv(y, sr, need, n_fft=512),
            "pv1024": stretch_pv(y, sr, need, n_fft=1024),
            "sola30": stretch_sola(y, sr, need),
            "resamp": stretch_resample(y, sr, need),
        }
        for name, y2 in cands.items():
            got = len(y2) / sr / nsyl
            heard = asr.transcribe(rec, y2.astype("float32"), sr)
            ok = hl or pinyin_gate(exp, asr.norm(heard))
            m = methods[name]
            m[0] += ok; m[1] += 1; m[2].append(spectral_dist(y, y2, sr))
            sf.write(os.path.join(args.out, f"{uid}_{name}.wav"), y2, sr, "PCM_16")
        if idx < 3:
            print(f"  样例 {texts[uid][:22]!r} 未校 {rate:.3f} 秒/音节 → 需拉长 {need:.3f}×")

    print(f"\n{'方法':<8}{'ASR 通过':>10}{'频谱距离(越小越像原声)':>24}")
    for name, (ok, tot, ds) in sorted(methods.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:<8}{ok:>6}/{tot:<3} {np.mean(ds):>12.4f} ± {np.std(ds):.4f}")
    print(f"\n同批未校原声的 ASR 通过率是生产门保证的 ~100%；"
          f"产物在 {args.out}/<id>_<方法>.wav，可人耳抽查")


if __name__ == "__main__":
    main()
