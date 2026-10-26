#!/usr/bin/env python3
"""给**已经在设备上合成好的 wav** 打分，口径与 scripts/ruler.py 完全一致。

为什么需要：本仓所有错率都是"Mac 上合成 + Mac 上 ASR 回读"。但端侧 runtime 与我们的复刻
在**含标点的句子**上行为不等价（真机取证：`。，、！？` 会被拆成两次独立推理，`;` 不会），
所以"改标点能降错率"这个结论**必须在设备产出的音频上重测**，不能拿 Mac 的数搬过去。

清单格式（TSV，`#` 开头行忽略，列间 TAB）：
    期望文本 <TAB> wav 路径 [<TAB> 类别]
用法：
    python scripts/score_wavs.py --manifest out/device/C05.tsv
同一份清单跑一次即可；要比两个条件就各写一份清单（期望列相同，只有路径不同）。
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr
from ruler import error_modes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--norm", type=float, default=0.0,
                    help="回读前把峰值归一到该值（0=不归一）。设备音量链路与 Mac 不同，"
                         "对比两端时要显式说明这一项是否开着")
    args = ap.parse_args()

    rows = []
    for line in open(args.manifest, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            rows.append((p[0], p[1], p[2] if len(p) > 2 else "未分类"))
    missing = [w for _, w, _ in rows if not os.path.exists(w)]
    if missing:
        for m in missing[:5]:
            print(f"!! 缺文件 {m}")
        raise SystemExit(f"清单里 {len(missing)}/{len(rows)} 个 wav 不存在")

    import soundfile as sf
    rec = asr.load_recognizer(args.threads)
    tot_err = tot_err_py = tot_n = 0
    mode_tot = collections.Counter()
    cat = collections.defaultdict(lambda: [0, 0])
    for exp, path, tag in rows:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if args.norm:
            peak = float(abs(audio).max())
            if peak > 1e-9:
                audio = audio * (args.norm / peak)
        heard = asr.transcribe(rec, audio, sr)
        expn, expp = asr.norm(exp), asr.py(exp)
        hp = asr.py(heard)
        e = 0 if asr.norm(heard) == expn else 1
        ep = 0 if hp == expp else 1
        modes = collections.Counter()
        if ep:
            for k, v in zip(("复制", "插音", "吞并", "替换"), error_modes(expp, hp)):
                modes[k] += v
        tot_err += e; tot_err_py += ep; tot_n += 1
        mode_tot.update(modes)
        cat[tag][0] += 1; cat[tag][1] += ep
        mark = "OK " if not ep else "ERR"
        print(f"[{mark} {tag}] {os.path.basename(path)}  {round(len(audio)/sr, 2)}s")
        print(f"      期望 {exp}\n      听到 {heard}")
    if tot_n:
        print(f"\nTOTAL n={tot_n}  字口径 err={tot_err} = {tot_err/tot_n*100:.1f}%"
              f"   音口径 err={tot_err_py} = {tot_err_py/tot_n*100:.1f}%")
        print("形态合计(按音节):", " ".join(f"{k}={mode_tot[k]}"
                                          for k in ("复制", "插音", "吞并", "替换")))
        print("分类别音口径:", "  ".join(f"{k}={v[1]/v[0]*100:.0f}%(n={v[0]})"
                                      for k, v in sorted(cat.items(), key=lambda kv: -kv[1][1]/kv[1][0])))


if __name__ == "__main__":
    main()
