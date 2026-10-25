#!/usr/bin/env python3
"""把教师语料的「语速随句长下降」曲线抹平：按句拉慢，不改音高、不改文本。

为什么要它（2026-10-25 实测定位的机制）：CosyVoice2 老师长句明显赶——
7~9 音节 0.285 秒/音节（≈V1 自己的 0.284），13~15 音节 0.255，16+ 音节 0.240~0.248，
而 >10 音节的句子占了 59% 的音节量。duration 头学到的就是"句子一长就要快"，
导出物每条输出中位只有 V1 时长的 0.87（正好等于长句那 12~15% 的缺口），
音节被挤出窗外 → 表现为吞字。所以先修**目标值**，不是调 lr。

口径为什么用"整段 wav 时长"而不是裁掉静音：训练标签就是总帧数 ÷ token 数，
V1 侧我量的也是合成音频总长，两边同口径才可比；裁静音会把数算小、把该拉的都漏掉。

⚠ 只朝一个方向拉（只拉快、不压慢）会把**整个分布**往上推：实测短句档 7~9 音节本来就在目标
0.285，做完后变 0.300——因为那档里"比目标快"的个体被拉了、"比目标慢"的原样留着，
非对称截断把均值抬上去了。长句档（真正要治的）落在 0.283。接受这个偏慢是有理由的：
我们治的就是"压太快"，往慢的方向偏 2% 比往快的方向偏安全；但它确实存在，别当成"完全抹平"。

用法：
  python scripts/flatten_rate.py --corpus out/corpus1k --dry-run   # 只看计划，不落盘
  python scripts/flatten_rate.py --corpus out/corpus1k --out out/corpus1k_flat
产物是**新目录**（wav/ + transcripts.tsv 同布局），原语料不动，可随时回退。
"""
import argparse
import collections
import os
import shutil

import librosa
import numpy as np
import soundfile as sf
from pypinyin import lazy_pinyin, Style


def n_syllables(text):
    return len(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True,
                           errors="default"))


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


def table(rows):
    """rows: [(uid, syl, dur)] -> 按音节区间分桶的 秒/音节 表"""
    b = collections.defaultdict(list)
    for _, syl, dur in rows:
        b[(syl - 1) // 3 * 3 + 1].append((syl, dur))
    out = []
    for lo in sorted(b):
        st = sum(s for s, _ in b[lo]); sd = sum(d for _, d in b[lo])
        out.append((lo, len(b[lo]), st, sd / st))
    return out


def print_table(name, rows):
    print(f"\n== {name} ==")
    print(f"{'音节区间':<9}{'句数':>6}{'总音节':>8}{'秒/音节':>10}")
    for lo, n, st, rate in table(rows):
        print(f"{lo}-{lo+2:<6}{n:>6}{st:>8}{rate:>10.3f}")
    st = sum(s for _, s, _ in rows); sd = sum(d for _, _, d in rows)
    print(f"{'全体':<9}{len(rows):>6}{st:>8}{sd / st:>10.3f}")


def verify(new_dir, ids, texts, threads=4, prune=False, orig_lines=None):
    """ASR 回读拉伸后的每一句，用**语料生产时同一个门**（gen_teacher_corpus 的
    `has_letter or pinyin_gate`）。拉伸可能引入两类失败：相位声码器的瞬态糊（伪影），
    以及"没拉够、句子仍然太快"（ASR 自己在快语上丢音节）——两类都在实测里出现过。
    所以这个工具必须自己复验并剔除，否则等于拿音质换语速，训练照样坏。
    prune=True 时把失败的 wav 移进 <new_dir>/rejected/ 并从 transcripts.tsv 摘掉。"""
    import asr
    from gen_teacher_corpus import _HAS_LETTER, pinyin_gate
    rec = asr.load_recognizer(threads)
    ok, failed = 0, []
    for uid in ids:
        exp = asr.norm(texts[uid])
        has_letter = bool(_HAS_LETTER.search(texts[uid]))
        y, sr = sf.read(os.path.join(new_dir, "wav", uid + ".wav"), dtype="float32")
        heard = asr.transcribe(rec, y, sr)
        if has_letter or pinyin_gate(exp, asr.norm(heard)):
            ok += 1
        else:
            failed.append(uid)
            print(f"    [复验失败] {uid} {texts[uid]!r} 回读={heard!r}")
    print(f"\n== ASR 复验 {len(ids)} 句 ==")
    print(f"  通过 {ok}/{len(ids)} = {ok/len(ids)*100:.1f}%（生产时的门）")
    if prune and failed:
        rej = os.path.join(new_dir, "rejected")
        os.makedirs(rej, exist_ok=True)
        for uid in failed:
            shutil.move(os.path.join(new_dir, "wav", uid + ".wav"),
                        os.path.join(rej, uid + ".wav"))
        drop = set(failed)
        lines = orig_lines if orig_lines is not None else [
            f"{u}\t{t}" for u, t in texts.items()]
        with open(os.path.join(new_dir, "transcripts.tsv"), "w", encoding="utf-8") as f:
            kept = 0
            for line in lines:
                if line.split("\t")[0].strip() not in drop:
                    f.write(line.rstrip("\n") + "\n"); kept += 1
        print(f"  已剔除 {len(failed)} 句 → {rej}/，transcripts.tsv 剩 {kept} 行")
    return ok, failed


def stretch_pv(y, factor, n_fft=512, hop=128):
    """相位声码器拉伸。窗长是关键：librosa 默认 n_fft=2048（16k 下 128ms）会把辅音瞬态
    糊掉，实测同批极端句 1.25× 截断拉伸只过 59/80；换 512 后不截拉伸量能过 39/40。"""
    return librosa.effects.time_stretch(y, rate=1.0 / factor, n_fft=n_fft, hop_length=hop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="含 wav/ 与 transcripts.tsv 的语料目录")
    ap.add_argument("--out", default="", help="输出目录（默认 <corpus>_flat）")
    ap.add_argument("--target-rate", type=float, default=0.0,
                    help="目标 秒/音节；0 = 取语料短句(5~10音节)那档的加权值")
    ap.add_argument("--max-stretch", type=float, default=1.6,
                    help="单句最多拉长倍率。**别学第一次截到 1.25**：截断后句子仍然太快，"
                         "ASR 会自己丢音节，看起来像拉伸伪影（选型数据见 rate_correction_lab.py）")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="偏差小于这个比例就不动（免得起无谓的伪影）")
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--regate", type=int, default=1,
                    help="1 = 产出后用生产门全库复验并剔除新失败句（拉伸的代价必须自己收口）")
    ap.add_argument("--verify", type=int, default=0, help="只复验拉伸幅度最大的 N 句")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = args.out or args.corpus.rstrip("/") + "_flat"
    texts = load(args.corpus)
    orig_lines = [l.rstrip("\n") for l in open(os.path.join(args.corpus, "transcripts.tsv"),
                                                encoding="utf-8")
                  if l.strip() and not l.startswith("#")]
    rows, audio = [], {}
    for uid, text in texts.items():
        path = os.path.join(args.corpus, "wav", uid + ".wav")
        if not os.path.isfile(path):
            continue
        syl = n_syllables(text)
        if not syl:
            continue
        y, sr = librosa.load(path, sr=None, mono=True)
        d = librosa.get_duration(y=y, sr=sr)
        rows.append((uid, syl, d)); audio[uid] = (y, sr, syl, d)
    if not rows:
        raise SystemExit(f"{args.corpus} 里没有可用的 wav+转写")

    short = [(s, d) for _, s, d in rows if 5 <= s <= 10]
    target = args.target_rate or (sum(d for _, d in short) / sum(s for s, _ in short))
    print(f"语料 {len(rows)} 句 / {sum(s for _, s, _ in rows)} 音节；"
          f"目标 {target:.3f} 秒/音节（{'命令行给定' if args.target_rate else '取 5~10 音节档实测'}）")
    print_table("校正前", rows)

    plan = []
    for uid, syl, d in rows:
        need = target / (d / syl)                     # >1 = 要拉长
        f = min(need, args.max_stretch) if need > 1 else 1.0
        plan.append((uid, f, need))
    n_touch = sum(1 for _, f, _ in plan if abs(f - 1) > args.tol)
    n_cap = sum(1 for _, _, need in plan if need > args.max_stretch + args.tol)
    print(f"\n要动 {n_touch} 句；其中 {n_cap} 句需求超过 --max-stretch "
          f"{args.max_stretch}，按上限截（截过的句仍会偏快，看校正后表）")
    if args.dry_run:
        print("[dry-run] 未落盘")
        return

    os.makedirs(os.path.join(out, "wav"), exist_ok=True)
    fmap = {uid: f for uid, f, _ in plan}
    after, skipped = [], 0
    for uid, syl, d in rows:
        y, sr, _, _ = audio[uid]
        f = fmap[uid]
        if abs(f - 1) <= args.tol:
            shutil.copy2(os.path.join(args.corpus, "wav", uid + ".wav"),
                         os.path.join(out, "wav", uid + ".wav"))
            skipped += 1
            after.append((uid, syl, d))
            continue
        y2 = stretch_pv(y, f, args.n_fft, args.hop)
        sf.write(os.path.join(out, "wav", uid + ".wav"), y2, sr, "PCM_16")
        after.append((uid, syl, len(y2) / sr))
    shutil.copy2(os.path.join(args.corpus, "transcripts.tsv"),
                 os.path.join(out, "transcripts.tsv"))
    print(f"已写 {out}（原样复制 {skipped} 句，拉伸 {len(rows) - skipped} 句，"
          f"n_fft={args.n_fft} hop={args.hop}）")

    ids = [uid for uid, _, _ in rows]
    if args.verify:
        ids = [uid for uid, f, _ in sorted(plan, key=lambda t: -t[1])[:args.verify]]
    if args.regate or args.verify:
        ok, failed = verify(out, ids, texts, args.threads, prune=bool(args.regate),
                            orig_lines=orig_lines)
        pruned = set(failed)
        after = [r for r in after if r[0] not in pruned]
        print(f"保留 {len(after)} 句 / {sum(s for _, s, _ in after)} 音节 "
              f"≈ {sum(d for _, _, d in after) / 3600:.2f} 小时")

    print_table("最终（复验并剔除后）", after)

    def long_dev(rows_):
        band = [r for lo, _, _, r in table(rows_) if lo >= 13]
        return max(abs(r - target) / target for r in band) * 100 if band else 0.0

    dev = long_dev(after)
    print(f"\n长句档(≥13 音节)相对目标最大偏差：校正前 {long_dev(rows):.1f}% → 最终 {dev:.1f}%"
          f"  →  {'抹平（<3% 判据通过）' if dev < 3 else '还没抹平，看上面最终表'}")


if __name__ == "__main__":
    main()
