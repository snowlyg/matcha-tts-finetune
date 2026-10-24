#!/usr/bin/env python3
"""把作者 V1 权重转成 icefall train.py 可续训的 exp-dir/epoch-1.pt。

取证结论（2026-10-24）：
- 必须用 pytorch_model_sherpa.bin：其 embedding 与现网 model-steps-3.onnx 逐元素相同；
  pytorch_model.bin 是另一份权重（303/305 个张量不同），不要拿来做端侧回落基线。
- 键名带 `model.` 前缀（作者保存的是 wrapper 的 state_dict）；icefall MatchaTTS
  顶层键是 encoder.*/decoder.*/mel_mean/mel_std —— 本脚本负责剥前缀。
- 作者 emb 行数 = 2190（tokens.txt 无 `_`，id 1..2190，其中 id2190 从未被喂、
  是潜伏越界）。配 `_ 0` 词表后 n_vocab=2191，本脚本把 emb 扩到 2191 行：
  行 0..2189 原样保留（行 0 作者侧本就是没用过的死行，现作 `_` padding，
  被 x_mask 遮掉），新增行 2190 用 TextEncoder 同款 normal_(0, 192^-0.5) 初始化
  （`<|unused_2190|>` 推理从不喂入，取值无影响）。
- --mel-mean/--mel-std 可用自算 cmvn 覆写 buffer，保证训练归一化域一致
  （V1 是 16k，公共 MatchaFbank 配置对不上其 buffer，详见 docs）。

用法（GPU 机，icefall venv）：
  python scripts/make_warmstart_ckpt.py \
    --bin models/zh-en/pytorch_model_sherpa.bin \
    --tokens data/tokens.txt \
    --out matcha/exp-1/epoch-1.pt [--mel-mean M --mel-std S]
之后 train.py 用 --start-epoch 2 续训。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_tokens(path):
    tok2id = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            info = line.rstrip("\n").split()
            if len(info) == 1:
                tok2id[" "] = int(info[0])
            else:
                tok2id[info[0]] = int(info[1])
    return tok2id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="models/zh-en/pytorch_model_sherpa.bin")
    ap.add_argument("--tokens", default="data/tokens.txt")
    ap.add_argument("--out", default="matcha/exp-1/epoch-1.pt")
    ap.add_argument("--mel-mean", type=float, default=None)
    ap.add_argument("--mel-std", type=float, default=None)
    args = ap.parse_args()

    sd = torch.load(args.bin, map_location="cpu", weights_only=False)
    keys = list(sd)
    if keys and keys[0] == "model":
        raise SystemExit("这份 bin 是 {'model': state_dict} 嵌套格式， unexpected")
    assert all(k.startswith("model.") for k in keys), "键名不全是 model.* 前缀"
    sd = {k[len("model."):]: v for k, v in sd.items()}

    tok2id = load_tokens(args.tokens)
    n_vocab = len(tok2id)
    max_id = max(tok2id.values())
    assert max_id == n_vocab - 1, (max_id, n_vocab)  # ids 0..n_vocab-1

    emb = sd["encoder.emb.weight"]
    rows, dim = emb.shape
    if rows < n_vocab:
        assert rows == n_vocab - 1, (rows, n_vocab)
        new = torch.empty(1, dim)
        torch.nn.init.normal_(new, 0.0, dim ** -0.5)
        sd["encoder.emb.weight"] = torch.cat([emb, new], dim=0)
        print(f"emb: {rows} -> {n_vocab} 行")
    else:
        assert rows == n_vocab, (rows, n_vocab)
    if args.mel_mean is not None:
        sd["mel_mean"] = torch.tensor(float(args.mel_mean))
    if args.mel_std is not None:
        sd["mel_std"] = torch.tensor(float(args.mel_std))
    print(f"mel_mean={float(sd['mel_mean']):.4f} mel_std={float(sd['mel_std']):.4f}")

    ckpt = {
        "model": sd,
        "model_avg": dict(sd),
        "optimizer": None,
        "scheduler": None,
        "grad_scaler": None,
        "sampler": None,
        "epoch": 1,
        "step": 0,
        "best_train_epoch": 1,
        "best_valid_epoch": 1,
        "batch_idx_train": 0,
        "best_train_loss": float("inf"),
        "best_valid_loss": float("inf"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"wrote {args.out} (train.py 用 --start-epoch 2 从这里续)")


if __name__ == "__main__":
    main()
