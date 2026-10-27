#!/usr/bin/env python3
"""客观停顿测量：不靠耳朵判"有没有停顿"，用帧能量算。

口径（与设备侧验收对齐，别改）：10ms 一帧，帧 RMS，阈值 = max(1e-5, 0.02 × 全句峰值)，
裁掉首尾静音得到语音段，语音段内**连续低于阈值的最长一段**就是句内最长静音。

为什么要它：`；`/`;` 会加 token 但不拆段，耳朵很难判它到底停没停；空格/换行则完全无效。
这两种"看起来像边界其实不是"的写法，只有量出来才说得清（结论见 docs/device-runtime-check.md）。

配对约定：文件名 `<base><a|b>[_r<N>].wav`，同一 base（同一 rep）的 a/b 为一对。
用法：
    python scripts/device/measure_pause.py --dir out/device/punct
    python scripts/device/measure_pause.py --dir out/device/rep --glob 'D0*'
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402

FRAME_S = 0.01
PEAK_RATIO = 0.02
FLOOR = 1e-5
NAME = re.compile(r"^(?P<base>.+?)(?P<cond>[ab])(?:_r(?P<rep>\d+))?$")


def measure(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(round(sr * FRAME_S))
    n = len(x) // hop
    if n == 0:
        return dict(total=0.0, speech=0.0, longest=0.0, lead=0.0, trail=0.0, peak=0.0)
    frames = np.array([float(np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2))) for i in range(n)])
    peak = float(np.max(np.abs(x)))
    thr = max(FLOOR, PEAK_RATIO * peak)
    voiced = frames >= thr
    total = len(x) / sr
    if not voiced.any():
        return dict(total=total, speech=0.0, longest=0.0, lead=total, trail=0.0, peak=peak)
    i0 = int(np.argmax(voiced))
    i1 = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    run = best = 0
    for k in range(i0, i1 + 1):
        if not voiced[k]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return dict(
        total=total,
        speech=(i1 - i0 + 1) * hop / sr,
        longest=best * hop / sr,
        lead=i0 * hop / sr,
        trail=(len(voiced) - 1 - i1) * hop / sr,
        peak=peak,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(config.OUT, "device"),
                    help="wav 目录（默认 out/device，out/ 不入库）")
    ap.add_argument("--glob", default="*.wav", help="文件名过滤，如 'D0*'")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not paths:
        raise SystemExit(f"{args.dir}/{args.glob} 没有 wav")

    rows, pairs = {}, {}
    for p in paths:
        stem = os.path.basename(p)[:-4]
        rows[stem] = measure(p)
        m = NAME.match(stem)
        if m:
            pairs.setdefault((m.group("base"), m.group("rep") or ""), {})[m.group("cond")] = stem

    print(f"{'case':16s} {'整段s':>8s} {'语音段s':>8s} {'句内最长静音s':>14s} "
          f"{'首静音s':>8s} {'尾静音s':>8s} {'峰值':>8s}")
    for k in sorted(rows):
        r = rows[k]
        print(f"{k:16s} {r['total']:8.3f} {r['speech']:8.3f} {r['longest']:14.3f} "
              f"{r['lead']:8.3f} {r['trail']:8.3f} {r['peak']:8.4f}")

    print("\n=== A/B 配对（同句、同参数，只差一个边界符号） ===")
    print("整段=含首尾静音的音频长度；语音段=裁掉首尾静音；句内最长静音=这条播报里最长的一次停顿")
    for key in sorted(pairs):
        g = pairs[key]
        if "a" not in g or "b" not in g:
            continue
        a, b = rows[g["a"]], rows[g["b"]]
        label = f"{key[0]}" + (f" r{key[1]}" if key[1] else "")
        print(f"{label:12s} 整段 {a['total']:.3f}->{b['total']:.3f} ({b['total'] - a['total']:+.3f}) | "
              f"语音段 {a['speech']:.3f}->{b['speech']:.3f} | "
              f"句内最长静音 {a['longest']:.3f}->{b['longest']:.3f} ({b['longest'] - a['longest']:+.3f})")


if __name__ == "__main__":
    main()
