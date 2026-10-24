#!/usr/bin/env python3
"""训练数据准备：把 (wav, 转写) 语料整理成 Lhotse CutSet 清单。

输入语料目录（二选一）：
  corpus/wav/<id>.wav + corpus/transcripts.tsv   （id<TAB>转写[<TAB>类别]，只取前两列）
  corpus/wav/<id>.wav + corpus/<id>.txt

输出 out/fbank/：baker_zh_cuts_{train,valid,test}.jsonl.gz（带 supervisions[0].text）
+ tokens_{split}.jsonl.gz（g2p 产物，旁证）+ summary.txt。

注意：这只是"原始 cuts"。训练前还要在 icefall 环境跑
local/convert_text_to_tokens.py（jieba+lazy_pinyin(tone_sandhi)+piper 写 c.tokens）
和 local/compute_fbank_baker_zh.py（算特征+cmvn）。见 docs/icefall-finetune-flow.md。
"""
import argparse
import gzip
import json
import os
import random

import soundfile as sf

import config
import g2p


def load_corpus(corpus):
    wav_dir = os.path.join(corpus, "wav")
    tsv = os.path.join(corpus, "transcripts.tsv")
    texts = {}
    if os.path.isfile(tsv):
        for line in open(tsv, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) >= 2:
                texts[p[0].strip()] = p[1].strip()
    else:
        for fn in os.listdir(wav_dir):
            if fn.endswith(".wav"):
                uid = fn[:-4]
                tp = os.path.join(corpus, uid + ".txt")
                if os.path.isfile(tp):
                    texts[uid] = open(tp, encoding="utf-8").read().strip()
    recs = []
    for uid in sorted(texts):
        wp = os.path.join(wav_dir, uid + ".wav")
        if not os.path.isfile(wp):
            print(f"[skip] no wav: {uid}")
            continue
        info = sf.info(wp)
        recs.append({"id": uid, "wav": os.path.abspath(wp), "text": texts[uid],
                     "sr": info.samplerate, "duration": info.duration})
    return recs


def build_cuts_lhotse(recs):
    from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
    cuts = []
    for r in recs:
        rec = Recording.from_file(r["wav"], recording_id=r["id"])
        sup = SupervisionSegment(id=r["id"], recording_id=r["id"], start=0.0,
                                 duration=r["duration"], text=r["text"])
        cuts.append(MonoCut(id=r["id"], start=0.0, duration=r["duration"],
                            channel=0, recording=rec, supervisions=[sup]))
    return CutSet.from_cuts(cuts)


def write_jsonl_gz(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default=os.path.join(config.OUT, "data"))
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--split", default="0.96,0.02,0.02")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--prefix", default="baker_zh_cuts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _, tok2id = g2p.load_vocab()
    recs = load_corpus(args.corpus)
    if not recs:
        raise SystemExit("no usable (wav,text) pairs found")
    bad = [r for r in recs if r["sr"] != args.sr]
    if bad:
        print(f"[warn] {len(bad)} wavs not {args.sr}Hz, skipped")
        recs = [r for r in recs if r["sr"] == args.sr]
    for r in recs:
        r["tokens"] = g2p.text_to_ids(r["text"], tok2id)
    recs = [r for r in recs if r["tokens"]]

    rnd = random.Random(args.seed)
    rnd.shuffle(recs)
    tr, va, te = [float(x) for x in args.split.split(",")]
    n = len(recs)
    i_va, i_te = int(n * tr), int(n * (tr + va))
    splits = {"train": recs[:i_va], "valid": recs[i_va:i_te], "test": recs[i_te:]}

    fbank = os.path.join(args.out, "fbank")
    os.makedirs(fbank, exist_ok=True)
    lines = []
    for name, rows in splits.items():
        hours = sum(r["duration"] for r in rows) / 3600.0
        lines.append(f"{name:6s} n={len(rows):5d} hours={hours:6.2f}")
        if args.dry_run:
            continue
        write_jsonl_gz(os.path.join(fbank, f"tokens_{name}.jsonl.gz"),
                       [{"id": r["id"], "tokens": r["tokens"]} for r in rows])
        try:
            build_cuts_lhotse(rows).to_file(
                os.path.join(fbank, f"{args.prefix}_{name}.jsonl.gz"))
        except ImportError:
            print("[warn] lhotse 未装；写最小 jsonl.gz（icefall 需 lhotse 格式）")
            write_jsonl_gz(os.path.join(fbank, f"{args.prefix}_{name}.jsonl.gz"),
                           [{k: r[k] for k in ("id", "wav", "text", "duration")} for r in rows])
    print("\n".join(lines))
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"total n={n} hours={sum(r['duration'] for r in recs)/3600:.2f}")


if __name__ == "__main__":
    main()
