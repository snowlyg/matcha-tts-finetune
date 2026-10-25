#!/usr/bin/env python3
"""时长门控：候选 ONNX 与 V1 对同一串 token 的输出，分别量**语音时长**与**首尾静音**。

为什么必须分成两个量（2026-10-25 我自己量错过一次，记录在这里防再犯）：
这工具最早只比"整段音频长度"，据此我断言"微调把语速压快 13%，所以吞字"——**那是错的**。
把首尾静音裁掉再比，臂 A 的语音时长比是 1.000（音节根本没念快），差的 0.16 全在静音上：
V1 每条固定带 ~0.66 秒首尾静音，训练后只剩 ~0.32 秒。用整段长度当"语速"，量到的是静音。
所以两个量分开算、**判据只看语音那条**；静音单独报出来（它是会被训练改变、用户也听得出的
行为，但不是"语速"）。

用法：
  python scripts/duration_gate.py --cand out/ab/a --n 3
  python scripts/duration_gate.py --cand <候选目录> --ref models/zh-en --phrases-file scripts/ruler_heldout.tsv
判据（跑前登记）：语音时长中位比 ∈[0.95,1.05] 且无单条 <0.85 → 通过；静音差值只报不判。
"""
import argparse
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_phrases(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        feed = line.split("\t")[0].strip()
        if ";" in feed or "weigh" in feed:      # 字母/英文行的 token 数与时长关系不稳定
            continue
        out.append(feed)
    return out


def speech_span(y, sr, top_db=30, win=400):
    """能量门限找语音段起止秒（不用 librosa：实例上的 venv 不一定装了它）。

    win=400 采样 = 25ms（16k）。门限取"全句最强帧"的 10^(-top_db/20)，找第一个/最后一个过阈帧。
    """
    if len(y) < win:
        return 0.0, len(y) / sr
    frames = np.abs(y[:len(y) // win * win].reshape(-1, win)).mean(axis=1)
    thr = frames.max() * 10.0 ** (-top_db / 20.0)
    on = np.nonzero(frames > thr)[0]
    if len(on) == 0:
        return 0.0, 0.0
    return float(on[0] * win) / sr, float((on[-1] + 1) * win) / sr


def measure(model_dir, phrases, threads, n):
    """换模型目录要重新 import：config 在 import 时就把 TTS_MODEL_DIR 读成常量了。
    返回 {短语: (token 数, 整段秒数列表, 语音秒数列表)}——ODE 有噪声，只取一次不算结论。"""
    os.environ["TTS_MODEL_DIR"] = os.path.abspath(model_dir)
    for m in ("config", "g2p", "repro_v1"):
        sys.modules.pop(m, None)
    import repro_v1 as R
    am, voc, tok2id = R.build(threads)
    out = {}
    for p in phrases:
        full, speech, ntok = [], [], None
        for _ in range(n):
            ids, audio = R.synth(am, voc, tok2id, p)
            y = np.asarray(audio, dtype="float32")
            ntok = len(ids)
            full.append(len(y) / am.sample_rate)
            s0, s1 = speech_span(y, am.sample_rate)
            speech.append(s1 - s0)
        out[p] = (ntok, full, speech)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="候选 model-steps-3.onnx 所在目录")
    ap.add_argument("--ref", default=os.path.join(config.REPO, "models", "zh-en"),
                    help="参照目录（默认现网 V1）")
    ap.add_argument("--phrases-file", default=os.path.join(config.REPO, "scripts",
                                                           "ruler_heldout.tsv"))
    ap.add_argument("--n", type=int, default=3, help="每条短语采样次数")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    phrases = load_phrases(args.phrases_file)
    print(f"短语 {len(phrases)} 条 × {args.n} 次采样   候选={args.cand}  参照={args.ref}")
    ref = measure(args.ref, phrases, args.threads, args.n)
    cand = measure(args.cand, phrases, args.threads, args.n)

    sp_r, tot_r, pad_ref, pad_cand, worst = [], [], [], [], []
    print(f"\n{'短语':<26}{'token':>6}{'语音 参照':>10}{'候选':>8}{'比':>7}{'整段比':>8}"
          f"{'静音 参照→候选':>17}")
    for p in phrases:
        n1, f1, s1 = ref[p]; _, f2, s2 = cand[p]
        m1, m2 = statistics.mean(s1), statistics.mean(s2)
        t1, t2 = statistics.mean(f1), statistics.mean(f2)
        sp_r.append(m2 / m1); tot_r.append(t2 / t1)
        pad_ref.append(t1 - m1); pad_cand.append(t2 - m2)
        worst.append((m2 / m1, p))
        print(f"{p:<26}{n1:>6}{m1:>10.2f}{m2:>8.2f}{m2/m1:>7.2f}{t2/t1:>8.2f}"
              f"{t1-m1:>11.2f}s →{t2-m2:.2f}s")
    med = statistics.median(sp_r)
    lo = [p for r, p in worst if r < 0.85]
    print(f"\n语音时长比 中位 {med:.3f}   <0.85 的有 {len(lo)} 条"
          f"   整段时长比中位 {statistics.median(tot_r):.3f}"
          f"（两者差距＝静音贡献了多少）")
    print(f"每条固定静音：参照 {statistics.mean(pad_ref):.3f}s → 候选 {statistics.mean(pad_cand):.3f}s"
          f"   差 {statistics.mean(pad_cand) - statistics.mean(pad_ref):+.3f}s")
    for r, p in sorted(worst)[:3]:
        print(f"  语音最短 {r:.2f}  {p}")
    ok = 0.95 <= med <= 1.05 and not lo
    print(f"\n判据（跑前登记）：语音时长中位 ∈[0.95,1.05] 且无单条 <0.85  →  {'通过' if ok else '不通过'}")
    print("  注：本工具判时长与节奏，不判读音；错率用 ruler.py 的字/音双口径。")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
