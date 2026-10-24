#!/usr/bin/env python3
"""生成 icefall 训练用 data/tokens.txt：作者词表 + 首行 `_ 0`。

取证结论（2026-10-24，见 docs/author-model-facts.md）：
- V1 推理从不喂 `_`/blank（ONNX 元数据无 add_blank，sherpa-onnx 对 zh-en 跳过 BOS/EOS），
  所以 `_` 只做 batch padding，放 id 0 不改动作者 token 的任何 id（1..2190 原样保留）。
- icefall matcha/tokenizer.py 硬要求 token2id["_"] 与 token2id[" "]；作者词表空格已在 id1。
- id 必须钉死与设备端一致，导出的 ONNX 才能回落到现网运行时。

用法：
  python scripts/make_train_tokens.py --vocab models/vocab_tts.txt --out data/tokens.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=config.VOCAB,
                    help="作者词表：一行一个 token，id = 行号+1")
    ap.add_argument("--out", default="data/tokens.txt")
    args = ap.parse_args()

    with open(args.vocab, encoding="utf-8") as f:
        toks = [line.rstrip("\n") for line in f]
    toks = [t for t in toks if t != ""]  # 末行无换行的 splitlines 差异
    assert len(toks) == 2190, len(toks)
    assert len(set(toks)) == len(toks), "词表有重复"
    assert "_" not in toks
    for t in toks:
        assert t == t.strip() or t == " ", f"token 含首尾空白: {t!r}"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("_ 0\n")
        for i, t in enumerate(toks, 1):
            f.write(f"{t} {i}\n" if t != " " else f"{i}\n")
    print(f"wrote {args.out}: 1 pad(_) + {len(toks)} tokens, ids 0..{len(toks)}")


if __name__ == "__main__":
    main()
