#!/usr/bin/env python3
"""同机同线程的 RTF 对照：现网 Matcha zh-en(V1) vs 候选 ZipVoice，两边都走 sherpa-onnx python。

为什么在 PC 上量而不是只信上游公布的数字：端侧可行性要按"已知能跑的那颗模型同机外推"——
拿 V1 在这台机器、这个线程数下的 RTF 当分母，看候选是它的几倍，比拿别人机器上的绝对值可靠。
判据写死在这里：播报链路预算 RTF <= 2（一条播报约 2 秒音频，合成不能明显超过播放）。

两个必须同时报的数：
  steps —— ZipVoice 是流匹配，耗时随步数近似线性走，不写步数的 RTF 没意义；
  中位数与 p95 —— 尾差才是现场"卡一下"的来源。

准确率不在这里测：scripts/ruler.py --engine zipvoice 已经用同一把尺子覆盖，别在这里再写一套口径。

用法（ZipVoice 三个参数缺省时只出分母）：
    python scripts/zipvoice/pc_same_machine_rtf.py \\
        --zipvoice-dir <解压出来的 int8 包> \\
        --vocos-24k <vocos_24khz.onnx> \\
        --ref-wav <包内 test_wavs/xxx.wav> \\
        --ref-text <那段参考音频对应的文本>
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config  # noqa: E402


def load_texts(path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ruler import DEFAULT, load_phrases
    if path:
        return load_phrases(path)
    return [(f, e, t) for f, e, t in DEFAULT]


def bench(name, synth, texts, reps, steps):
    out = []
    for feed, _exp, _tag in texts:
        xs = []
        for _ in range(reps):
            t0 = time.monotonic()
            audio, sr = synth(feed)
            el = time.monotonic() - t0
            dur = len(audio) / sr if sr else 0.0
            if dur <= 0:
                print(f"!! {name} 空音频: {feed}")
                continue
            xs.append(el / dur)
        if xs:
            out.extend(xs)
            print(f"[{name}] {feed[:22]:22s} steps={steps} 中位RTF={statistics.median(xs):6.2f} "
                  f"最快={min(xs):.2f} 最慢={max(xs):.2f}")
    return out


def report(label, rtfs, budget=2.0):
    if not rtfs:
        print(f"{label}: 没有有效样本")
        return None
    s = sorted(rtfs)
    p95 = s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]
    med = statistics.median(s)
    verdict = "可行" if med <= budget else "判死"
    print(f"{label}: n={len(s)} 中位={med:.2f} p95={p95:.2f} 最大={s[-1]:.2f} "
          f"| 预算 RTF<={budget} -> {verdict}")
    return med


def build_matcha(threads):
    """与生产 shim 同契约：noise_scale=1.0 + 三个 -zh rule-fst + silence_scale=0.2 + 不批量合句。"""
    import sherpa_onnx as so
    md = config.MODEL_DIR
    m = so.OfflineTtsMatchaModelConfig(
        acoustic_model=config.acoustic_path(),
        vocoder=config.vocoder_path(),
        lexicon=os.path.join(md, "lexicon.txt"),
        tokens=os.path.join(md, "tokens.txt"),
        data_dir=os.path.join(md, "espeak-ng-data"),
        noise_scale=1.0)
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(matcha=m, num_threads=threads, provider="cpu"),
        rule_fsts=",".join(os.path.join(md, n) for n in
                           ("number-zh.fst", "date-zh.fst", "phone-zh.fst")),
        max_num_sentences=1, silence_scale=0.2)
    if not cfg.validate():
        raise SystemExit("Matcha config 校验失败：检查 models/ 下文件齐不齐（TTS_MODEL_DIR 指对了吗）")
    tts = so.OfflineTts(cfg)

    def synth(text):
        res = tts.generate(text)
        return np.asarray(res.samples, dtype="float32"), tts.sample_rate
    return synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4, help="现网 shim 写死 4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--phrases-file", default="")
    ap.add_argument("--zipvoice-dir", default=os.environ.get("ZV_MODEL_DIR", ""))
    ap.add_argument("--vocos-24k", default=os.environ.get("ZV_VOCOS", ""))
    ap.add_argument("--ref-wav", default=os.environ.get("ZV_REF_WAV", ""))
    ap.add_argument("--ref-text", default=os.environ.get("ZV_REF_TEXT", ""))
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()

    texts = load_texts(args.phrases_file)
    print(f"== 分母：Matcha V1 (threads={args.threads}, noise=1.0, 3 个 -zh fst) ==")
    mmed = report("Matcha", bench("matcha", build_matcha(args.threads), texts, args.reps, "-"))

    if not (args.zipvoice_dir and args.ref_wav and args.ref_text):
        print("\n没给 --zipvoice-dir/--ref-wav/--ref-text，只出分母。")
        return

    from ruler import build_zipvoice
    zv = args.zipvoice_dir
    if args.vocos_24k:
        dst = os.path.join(zv, "vocos_24khz.onnx")
        if not os.path.exists(dst):
            # build_zipvoice 只在模型目录里找 vocoder，24k 那份是单独下载的，软过去
            os.symlink(os.path.abspath(args.vocos_24k), dst)
            print(f"已把 --vocos-24k 软链到 {dst}")
    print(f"== 候选：ZipVoice (threads={args.threads}, steps={args.steps}) ==")
    synth, _sr = build_zipvoice(zv, args.ref_wav, args.ref_text, args.threads)
    zmed = report("ZipVoice", bench("zipvoice", synth, texts, args.reps, args.steps))

    if mmed and zmed:
        print(f"\n同机比值 ZipVoice/Matcha = {zmed / mmed:.1f}x —— 端侧能不能用就看这个数")


if __name__ == "__main__":
    main()
