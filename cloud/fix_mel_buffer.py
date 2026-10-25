#!/usr/bin/env python3
"""把训练 ckpt 里的 mel 归一化 buffer 换成训练实际用的那对 cmvn。

为什么需要：matcha/train.py 的 prepare_input 用 `--cmvn` 文件里的 mel_mean/mel_std
归一化特征，而模型反归一化用的是**自身 register_buffer**，热启动时那对 buffer 来自 V1。
两边不同一对 → 导出物输出的是被仿射拉歪的 mel（实测 std 1.9×、均值 +2~3dB），
声码器拿到域外输入 → 时长/内容还在、音质塌。权重本身是在"我们那对"的 z 空间里学的，
所以只要把 buffer 改成训练实际用的那对，已训好的 ckpt 可以直接救，不必重训。

用法：
  python fix_mel_buffer.py --ckpt matcha/exp-1/epoch-300.pt \
      --out matcha/exp-melfix/epoch-300.pt \
      --mean -6.210153102874756 --std 2.9019434452056885
键名不存在时只列出候选并退出，不会静默改错东西。
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mean", type=float, required=True)
    ap.add_argument("--std", type=float, required=True)
    ap.add_argument("--key-mean", default="mel_mean")
    ap.add_argument("--key-std", default="mel_std")
    args = ap.parse_args()

    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    for k in (args.key_mean, args.key_std):
        if k not in sd:
            cand = [x for x in sd if any(t in x.lower() for t in ("mel", "mean", "std"))]
            print(f"[中止] state_dict 里没有键 {k}；含 mel/mean/std 的键：")
            for c in cand[:40]:
                v = sd[c]
                print(f"   {c}  shape={tuple(v.shape) if hasattr(v,'shape') else type(v)}")
            print("用 --key-mean/--key-std 指定正确键名再跑。")
            raise SystemExit(2)

    before = (float(sd[args.key_mean]), float(sd[args.key_std]))
    after = (args.mean, args.std)
    for container in ("model", "model_avg"):
        if isinstance(obj, dict) and container in obj and isinstance(obj[container], dict):
            obj[container][args.key_mean] = torch.tensor(after[0])
            obj[container][args.key_std] = torch.tensor(after[1])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(obj, args.out)
    print(f"{args.ckpt}: ({before[0]:.4f}, {before[1]:.4f}) -> ({after[0]:.4f}, {after[1]:.4f})")
    print(f"wrote {args.out}  (顶层键: {list(obj)[:6]})")


if __name__ == "__main__":
    main()
