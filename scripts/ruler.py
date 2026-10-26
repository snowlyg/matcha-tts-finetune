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

import numpy as np

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



def build_zipvoice(model_dir, ref_wav, ref_text, threads):
    """第二个引擎：sherpa-onnx 里的 ZipVoice（零样本克隆，24k 输出）。

    用它打分要记住这**换的是整个模型**：ZipVoice 自带中英前端（espeak-ng + 自己的
    lexicon），不是 V1 那套 jieba+pypinyin。所以它答的是"换基座会不会更准"，
    不是"换声学头会不会更准"；音色用 --ref-wav 钉成 V1 自己的声音，尽量少一个变量。
    返回 (synth, sample_rate)，与 matcha 路径同构，打分逻辑完全共用。
    """
    import sherpa_onnx as so
    import soundfile as sf

    def find(*names):
        for n in names:
            p = os.path.join(model_dir, n)
            if os.path.exists(p):
                return p
        raise SystemExit(f"模型目录里找不到 {names}（目录名或版本不对？）: {model_dir}")

    enc = find("encoder.onnx", "zipvoice.encoder.onnx", "text_encoder.onnx")
    dec = find("decoder.onnx", "zipvoice.decoder.onnx", "fm_decoder.onnx")
    voc = find("vocos_24khz.onnx", "vocos-24khz-univ.onnx", "vocos-16khz-univ.onnx")
    z = so.OfflineTtsZipvoiceModelConfig(
        tokens=find("tokens.txt", "zipvoice.tokens"), encoder=enc, decoder=dec, vocoder=voc,
        data_dir=os.path.join(model_dir, "espeak-ng-data"),
        lexicon=find("lexicon.txt", "lexicon.espeak-ng.txt"))
    cfg = so.OfflineTtsConfig(
        model=so.OfflineTtsModelConfig(zipvoice=z, num_threads=threads, provider="cpu"))
    tts = so.OfflineTts(cfg)
    audio, sr = sf.read(ref_wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    gc = so.GenerationConfig()
    gc.reference_audio = audio
    gc.reference_sample_rate = sr
    gc.reference_text = ref_text
    gc.sid = 0
    gc.speed = 1.0
    gc.num_steps = 4

    def synth(text):
        out = tts.generate(text, gc)
        return np.asarray(out.samples, dtype="float32"), tts.sample_rate
    return synth, tts.sample_rate


def norm_to(audio, peak):
    """只改增益、不改时长：给 ASR 一个和 V1 同响度的输入。

    为什么需要：微调导出物的输出比 V1 轻 ~32 dB（RMS 0.0026 vs 0.104），而 ASR 在
    过轻输入上会大量丢字——不扣除这一项，"错率"里混着"听不清"，不能归因给模型。
    """
    if not peak:
        return audio
    m = float(np.abs(audio).max())
    return audio if m < 1e-9 else audio * (peak / m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--phrases-file", default="")
    ap.add_argument("--sample-out", default="")
    ap.add_argument("--norm", type=float, default=0.0,
                    help="回读前把音频峰值归一到该值（0=不归一），用来分离\"太轻听不清\"与\"念错\"")
    ap.add_argument("--engine", choices=["matcha", "zipvoice"], default="matcha",
                    help="matcha=现网/候选 V1 契约；zipvoice=换整个模型的对照（自带前端）")
    ap.add_argument("--zipvoice-dir", default="")
    ap.add_argument("--ref-wav", default="out/ref_v1.wav")
    ap.add_argument("--ref-text-file", default="out/ref_v1.txt")
    args = ap.parse_args()

    rec = asr.load_recognizer(args.threads)
    if args.engine == "zipvoice":
        if not args.zipvoice_dir:
            raise SystemExit("--engine zipvoice 要带 --zipvoice-dir（解压出来的模型目录）")
        synth, sr = build_zipvoice(args.zipvoice_dir, args.ref_wav,
                                   open(args.ref_text_file, encoding="utf-8").read().strip(),
                                   args.threads)
    else:
        am, voc, tok2id = rp.build(args.threads)
        sr = am.sample_rate

        def synth(text):
            return rp.synth(am, voc, tok2id, text)[1]
    if sr != 16000:                      # ASR 与固定静音统计统一按 16k 口径，别让采样率混进对比
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(16000, sr)

        def _s(t, _f=synth, _g1=16000 // g, _g2=sr // g):
            a, _ = _f(t)
            return resample_poly(a, _g1, _g2).astype("float32")
        synth = _s
        sr = 16000
    print(f"[ruler] engine={args.engine} sample_rate={sr} n={args.n} "
          f"phrases={args.phrases_file or '默认5条'}")
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
            audio = synth(feed)
            if i == 0 and args.sample_out:
                import soundfile as sf
                sf.write(os.path.join(args.sample_out, rp.safe(feed) + ".wav"),
                         audio, sr, "PCM_16")
            heard = asr.transcribe(rec, norm_to(audio, args.norm), sr)
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
