#!/usr/bin/env python3
"""把参照 ONNX 的 metadata_props 整份覆盖到另一个 ONNX 上。

用途：icefall 导出的元数据键集合与现网那份不一致（language / voice / has_espeak /
jieba / model_author…），而 sherpa-onnx 的前端会读这些键。逐键判断"哪些会影响运行时"
既费神又容易漏，直接把现网那份整份盖上去——diff 变空集是构造性保证。

⚠ num_ode_steps 默认保留导出物自己的值：它描述的是**这张图**跑了几个 ODE 步，
盖成参照里的 3 会让 model-steps-5.onnx 的元数据说谎（键值与图不符）。
同理 --keep 可加任何"必须与图一致"的键。
"""
import argparse

import onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="参照（现网在跑的那份 onnx）")
    ap.add_argument("--src", required=True, help="icefall 导出物")
    ap.add_argument("--out", required=True, help="覆盖后写出路径（可与 --src 相同=原地）")
    ap.add_argument("--keep", action="append", default=["num_ode_steps"],
                    help="保留 --src 取值的键（默认 num_ode_steps）；可重复")
    args = ap.parse_args()

    ref = {p.key: p.value for p in onnx.load(args.ref).metadata_props}
    m = onnx.load(args.src)
    src = {p.key: p.value for p in m.metadata_props}

    final = dict(ref)
    for k in args.keep:
        if k in src:
            final[k] = src[k]
        elif k in ref:
            print(f"[warn] --src 没有键 {k}，沿用参照值 {ref[k]!r}")

    while len(m.metadata_props):
        m.metadata_props.pop()
    for k in sorted(final):
        p = m.metadata_props.add()
        p.key = k
        p.value = str(final[k])
    onnx.save(m, args.out)

    print(f"wrote {args.out}  ({len(final)} keys)")
    print("only-in-ref(added):  ", sorted(set(ref) - set(src)))
    print("only-in-src(dropped):", sorted(set(src) - set(ref) - set(args.keep)))
    changed = {k: (src[k], final[k]) for k in src if k in ref and src[k] != final[k]}
    print("value-changed:", changed or "{}")


if __name__ == "__main__":
    main()
