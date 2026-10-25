#!/usr/bin/env python3
"""批量合成语料 + SenseVoice 回读门控（老师模型/自采样的拒绝采样器）。

用途：录不到合适真人时，用合成产训练语料：
  - --backend cosyvoice：CosyVoice2 zero-shot（--ref-wav/--ref-text 定音色）；
    先用小 n 对照 ruler 基线验证老师更准，再放大产语料。
  - --backend cosyvoice3：CosyVoice3（FunAudioLLM/Fun-CosyVoice3-0.5B），同一套门控与
    产物格式；它多了**中文拼音 inpainting**（可显式指定多音字读法）。换不换老师不由我说，
    由同一份 60 句对照的**首试通过率**说话（CosyVoice2 实测首试 752/1000、n=8 收 975）。
  - --backend v1：现网 matcha ONNX 拒绝采样（音色不变；只能强化它偶尔念对的，
    修不了它从念不对的）。

门控：每条最多试 --n 次，SenseVoice 回读与目标文本"去调拼音"相等才收
（ASR 同音选字不误杀；丢字/错音节仍拦；字母行只认精确匹配并放宽保留）。
产出：--out/wav/<id>.wav + transcripts.tsv（可直接喂 prepare_data.py /
score_speaker.py），另写 report.log（首试通过率/最终收集率——即老师的内容错率）。
被门控拒的音频留在 --out/rejected/<id>.wav，供人耳复核"是念错还是 ASR 误判"。

注意：
  - 转写一律是"念法文本"（数字写汉字；字母行的回读易出 i see you 式假错，
    脚本对含字母行默认跳过门控、全部保留，靠人耳抽听）。
  - CosyVoice 输出 24k，自动重采样 16k（需 scipy）。
"""
import argparse
import collections
import os
import re
import sys

import numpy as np
import soundfile as sf
from pypinyin import lazy_pinyin, Style

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr
import config
import repro_v1 as rp

_HAS_LETTER = re.compile(r"[A-Za-z]")


def pinyin_gate(expected_norm, heard_norm):
    """门控比较：去调拼音序列相等即通过。

    ASR 同音选字（输液→书页、床→闯、稍候→稍后）不该拦；
    丢字/多字/错音节（患者李四→患者李）长度或音节不等，仍拦。
    尺子(ruler.py)仍用字符精确匹配，不受此影响。
    """
    if expected_norm == heard_norm:
        return True
    if _HAS_LETTER.search(expected_norm) or _HAS_LETTER.search(heard_norm):
        return False  # 字母/汉字混排，拼音比较不可靠，只认精确
    a = lazy_pinyin(expected_norm, style=Style.NORMAL)
    b = lazy_pinyin(heard_norm, style=Style.NORMAL)
    return len(a) == len(b) and a == b


def load_script(path):
    """record_script.tsv: id<TAB>文本<TAB>类别；跳过 # 注释。"""
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) >= 2:
            rows.append((p[0].strip(), p[1].strip(), p[2].strip() if len(p) > 2 else ""))
    return rows


class V1Backend:
    name = "v1"

    def __init__(self, threads):
        self.am, self.voc, self.tok2id = rp.build(threads)

    def __call__(self, text, attempt):
        _, audio = rp.synth(self.am, self.voc, self.tok2id, text)
        return np.asarray(audio, dtype=np.float32), 16000


class CosyVoiceBackend:
    """CosyVoice2 / CosyVoice3 老师（同一个源码仓，两个类）。

    3.0 与 2.0 的调用差异：类名不同，且 prompt 文本必须带 `<|endofprompt|>` 分隔
    （前段是风格指令，后段才是参考音频的逐字文本）——照 2.0 的写法直接喂会得到
    很怪的结果。这里只补一个中性指令，**是否真的更好由 60 句产出率门判定**，
    不靠我对 API 的理解。
    """
    name = "cosyvoice"

    def __init__(self, model_dir, ref_wav, ref_text, threads, v3=False):
        import torch  # noqa: F401  cosyvoice 依赖
        import cosyvoice.cli.cosyvoice as cv
        want = "CosyVoice3" if v3 else "CosyVoice2"
        klass = getattr(cv, want, None)
        if klass is None:
            sys.exit(f"本地 CosyVoice 源码里没有 {want} —— 老师仓库要更新到对应版本"
                     f"（有 {want} 的 tag），否则这轮实验换不了老师")
        self.name = "cosyvoice3" if v3 else "cosyvoice"
        self.v3 = v3
        self.torch = torch
        self.m = klass(model_dir, load_vllm=False)
        # frontend_zero_shot 内部自己 load_wav(prompt_wav, 24000/16000) → 要路径，传张量会
        # 报 TypeError: Invalid file: tensor(...)
        self.ref_wav = os.path.abspath(ref_wav)
        self.ref_text = ref_text
        if v3 and "<|endofprompt|>" not in ref_text:
            self.ref_text = "You are a helpful assistant.<|endofprompt|>" + ref_text

    def __call__(self, text, attempt):
        self.torch.manual_seed(20261024 + attempt)
        chunks = [out["tts_speech"] for out in self.m.inference_zero_shot(
            text, self.ref_text, self.ref_wav, stream=False)]
        audio = self.torch.cat(chunks).squeeze().numpy()
        return np.asarray(audio, dtype=np.float32), self.m.sample_rate


def resample16k(audio, sr):
    if sr == 16000:
        return audio
    from scipy.signal import resample_poly
    g = np.gcd(16000, sr)
    return resample_poly(audio, 16000 // g, sr // g).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["v1", "cosyvoice", "cosyvoice3"], default="v1")
    ap.add_argument("--script", default=os.path.join(config.REPO, "data", "record_script.tsv"))
    ap.add_argument("--out", default=os.path.join(config.OUT, "synth_corpus"))
    ap.add_argument("--n", type=int, default=8, help="每条最多尝试次数")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（验老师用）")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--model-dir", default="",
                    help="cosyvoice/cosyvoice3: 权重目录（3.0 用 FunAudioLLM/Fun-CosyVoice3-0.5B）")
    ap.add_argument("--ref-wav", default="", help="cosyvoice: 参考音色 wav")
    ap.add_argument("--ref-text", default="", help="cosyvoice: 参考音色对应文本")
    args = ap.parse_args()

    # 先校验输入再加载权重：CosyVoice 载入要 1-2 分钟，别让缺文件白等
    if not os.path.isfile(args.script):
        sys.exit(f"录制稿不存在: {args.script}")
    if args.backend.startswith("cosyvoice"):
        for label, p in (("--model-dir", args.model_dir), ("--ref-wav", args.ref_wav)):
            if not p or not os.path.exists(p):
                sys.exit(f"{label} 无效（文件/目录不存在）: {p!r}")
        if not args.ref_text:
            sys.exit("--ref-text 不能为空（zero-shot 需要参考音频的念法文本）")

    backend = (V1Backend(args.threads) if args.backend == "v1" else
               CosyVoiceBackend(args.model_dir, args.ref_wav, args.ref_text, args.threads,
                                v3=(args.backend == "cosyvoice3")))
    rec = asr.load_recognizer(args.threads)

    wav_dir = os.path.join(args.out, "wav")
    rej_dir = os.path.join(args.out, "rejected")   # 被门控拒的也留音频，否则无法人耳复核
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(rej_dir, exist_ok=True)
    rows = load_script(args.script)
    if args.limit:
        rows = rows[:args.limit]

    rep = []
    kept = 0
    tsv = open(os.path.join(args.out, "transcripts.tsv"), "w", encoding="utf-8")
    for uid, text, cat in rows:
        expn = asr.norm(text)
        has_letter = bool(_HAS_LETTER.search(text))
        heard_cnt = collections.Counter()
        accepted = None
        for attempt in range(1, args.n + 1):
            audio, sr = backend(text, attempt)
            heard = asr.transcribe(rec, resample16k(audio, sr))
            heard_cnt[heard] += 1
            ok = has_letter or pinyin_gate(expn, asr.norm(heard))
            if ok:
                accepted = (audio, sr, attempt, heard)
                break
        if accepted is None:
            rep.append(f"[FAIL n={args.n}] {uid} {text!r} 回读={heard_cnt.most_common(3)}")
            sf.write(os.path.join(rej_dir, f"{uid}.wav"),
                     resample16k(audio, sr), 16000, "PCM_16")
            continue
        audio, sr, attempt, heard = accepted
        wav16 = resample16k(audio, sr)
        wpath = os.path.join(wav_dir, f"{uid}.wav")
        sf.write(wpath, wav16, 16000, "PCM_16")
        tsv.write(f"{uid}\t{text}\t{cat}\n")
        kept += 1
        first = "首试" if attempt == 1 else f"第{attempt}次"
        tag = "(含字母,门控放宽)" if has_letter else ""
        rep.append(f"[OK {first}] {uid} {text!r} -> {os.path.basename(wpath)} "
                   f"{len(wav16)/16000:.2f}s {tag}")
    tsv.close()

    first_try = sum(1 for r in rep if r.startswith("[OK 首试]"))
    summary = (f"backend={backend.name} 条数={kept}/{len(rows)} "
               f"首试通过={first_try}/{len(rows)} "
               f"=> 内容错率上界≈{(len(rows)-first_try)/max(len(rows),1)*100:.1f}% "
               f"(n={args.n}; 字母行门控放宽)\n")
    out = "\n".join(rep) + "\n" + summary
    with open(os.path.join(args.out, "report.log"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"corpus 在 {args.out}/（wav/ + transcripts.tsv），可直接跑 prepare_data.py / score_speaker.py")


if __name__ == "__main__":
    main()
