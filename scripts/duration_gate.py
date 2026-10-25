#!/usr/bin/env python3
"""时长门控：候选 ONNX 对同一串 token 的输出时长 vs V1 的比值。

为什么要有它（2026-10-25 定位的机制）：首轮微调 mel 域修好之后仍然 83% 错率，真正的病灶是
**输出整体被压快**——25 条 held-out 短语上 ep300/V1 时长比中位 0.87，11/25 条 <0.85，
最差 0.63；每音节 0.284s→0.251s。音节被挤出窗外就表现为吞字
（`一零三床准备接诊`→"一零三床准备枕"）。这个量比 ASR 错率**更早更便宜**地暴露 duration 头塌了：
不用过 ASR、不用采样几十次，一次合成就能算。

判据（登记在跑之前，别事后编）：中位比 ∈ [0.95, 1.05] 且没有任何一条 < 0.85 → 通过。

用法：
  python scripts/duration_gate.py --cand out/melfix/gate300/model-steps-3.onnx
  python scripts/duration_gate.py --cand <候选> --ref models/zh-en --n 8 --phrases-file scripts/ruler_heldout.tsv
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import repro_v1 as rp


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


def measure(model_dir, phrases, threads, n):
    """换模型目录要重新 import：config 在 import 时就把 TTS_MODEL_DIR 读成常量了。
    返回 {短语: (token 数, n 次采样各自的秒数)}——ODE 有噪声，得看离散度而不是只取一次。"""
    os.environ["TTS_MODEL_DIR"] = os.path.abspath(model_dir)
    for m in ("config", "g2p", "repro_v1"):
        sys.modules.pop(m, None)
    import repro_v1 as R
    am, voc, tok2id = R.build(threads)
    out = {}
    for p in phrases:
        ds, ntok = [], None
        for _ in range(n):
            ids, audio = R.synth(am, voc, tok2id, p)
            ntok = len(ids)
            ds.append(len(audio) / am.sample_rate)
        out[p] = (ntok, ds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="候选 model-steps-3.onnx 所在目录")
    ap.add_argument("--ref", default=os.path.join(config.REPO, "models", "zh-en"),
                    help="参照目录（默认现网 V1）")
    ap.add_argument("--phrases-file", default=os.path.join(config.REPO, "scripts",
                                                           "ruler_heldout.tsv"))
    ap.add_argument("--n", type=int, default=4, help="每条短语采样次数")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    phrases = load_phrases(args.phrases_file)
    print(f"短语 {len(phrases)} 条 × {args.n} 次采样   候选={args.cand}  参照={args.ref}")
    ref = measure(args.ref, phrases, args.threads, args.n)
    cand = measure(args.cand, phrases, args.threads, args.n)

    ratios, rates = [], []
    worst = []
    print(f"\n{'短语':<26}{'token':>6}{'参照 s':>8}{'候选 s':>8}{'比值':>7}")
    for p in phrases:
        n1, d1 = ref[p]; _, d2 = cand[p]
        m1, m2 = statistics.mean(d1), statistics.mean(d2)
        r = m2 / m1
        ratios.append(r)
        rates.append((n1, m1, m2))
        worst.append((r, p))
        print(f"{p:<26}{n1:>6}{m1:>8.2f}{m2:>8.2f}{r:>7.2f}")
    med = statistics.median(ratios)
    per = (sum(m2 for _, _, m2 in rates) / sum(n for n, _, _ in rates)) / \
          (sum(m1 for _, m1, _ in rates) / sum(n for n, _, _ in rates))
    lo = [p for r, p in worst if r < 0.85]
    print(f"\n中位时长比 {med:.3f}   每音节时长比 {per:.3f}   <0.85 的有 {len(lo)} 条")
    for r, p in sorted(worst)[:3]:
        print(f"  最差 {r:.2f}  {p}")
    ok = 0.95 <= med <= 1.05 and not lo
    print(f"\n判据（跑前登记）：中位 ∈[0.95,1.05] 且无单条 <0.85  →  {'通过' if ok else '不通过'}")
    print("  注：这是**时长/节奏**门控，不判读音；读音错率仍用 ruler.py 的双口径。")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
