#!/usr/bin/env bash
# 一次粘贴跑完：取证 mel buffer 键名 → 用训练实际用的 cmvn 覆写 → 导出 → 存档 → 打包。
# 背景：train.py 的 prepare_input 用 --cmvn 那对做归一化，模型反归一化用自身 buffer（热启动
# 时来自 V1）。两对不同一对，导出物的 mel 就被仿射拉歪（实测 std 1.9×、均值 +2~3dB），
# 声码器吃到域外输入 → 时长/内容还在、音质塌。权重是在"我们那对"的 z 空间里学的，
# 所以改 buffer 就能救已训好的 ckpt，不必重训。
#
# 用法（实例上，在 icefall 的 TTS 目录下）：
#   bash /root/matcha-tts-finetune/cloud/session_fix_mel.sh            # 默认 60 180 300
#   bash /root/matcha-tts-finetune/cloud/session_fix_mel.sh 100 200 300
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
ICEFALL="${ICEFALL:-$HOME/icefall}"
TTS="${TTS:-$ICEFALL/egs/baker_zh/TTS}"
PY="${PY:-python3}"
EPOCHS="${*:-60 180 300}"

cd "$TTS" || { echo "!! 没有 $TTS"; exit 1; }
[ -f data/fbank/cmvn.json ] || { echo "!! 缺 data/fbank/cmvn.json"; exit 1; }

echo "== 0) 取证：ckpt 里 mel buffer 的真实键名与当前值（只读，不改东西）=="
FIRST="$(echo "$EPOCHS" | awk '{print $1}')"
[ -f "matcha/exp-1/epoch-$FIRST.pt" ] || FIRST="$(ls matcha/exp-1/ 2>/dev/null | sed -n 's/^epoch-\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1)"
echo "   探针用 epoch $FIRST"
"$PY" - "matcha/exp-1/epoch-$FIRST.pt" <<'PYEOF'
import sys, torch
o = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sd = o['model']
print('顶层键:', list(o)[:8])
hits = {k: float(v) for k, v in sd.items() if any(t in k.lower() for t in ('mel', 'mean', 'std'))
        and hasattr(v, 'numel') and v.numel() == 1}
print('标量 mel/mean/std 键:', hits)
PYEOF

echo "== 1) 训练实际用的那对 cmvn =="
MEAN="$("$PY" -c "import json;print(json.load(open('data/fbank/cmvn.json'))['fbank_mean'])")"
STD="$("$PY" -c "import json;print(json.load(open('data/fbank/cmvn.json'))['fbank_std'])")"
echo "   mel_mean=$MEAN mel_std=$STD"

mkdir -p matcha/exp-melfix cand
for e in $EPOCHS; do
  src="matcha/exp-1/epoch-$e.pt"
  [ -f "$src" ] || { echo "[skip] 没有 $src"; continue; }
  echo "== 2) 覆写 buffer：$src -> matcha/exp-melfix/epoch-$e.pt"
  "$PY" "$REPO/cloud/fix_mel_buffer.py" --ckpt "$src" \
    --out "matcha/exp-melfix/epoch-$e.pt" --mean "$MEAN" --std "$STD" || continue
  echo "== 3) 导出 epoch $e（导出物写在 CWD，导完立刻改名存档，防同名覆盖）"
  "$PY" matcha/export_onnx.py --exp-dir matcha/exp-melfix --epoch "$e" \
    --tokens data/tokens.txt --cmvn data/fbank/cmvn.json > "/tmp/export-melfix-$e.log" 2>&1 \
    && cp model-steps-3.onnx "cand/steps-3-melfix-ep$e.onnx" \
    && ls -lh "cand/steps-3-melfix-ep$e.onnx" \
    || echo "[warn] epoch $e 导出失败，看 /tmp/export-melfix-$e.log"
done

echo "== 4) 校验导出物里的 mel 常数是否已换过来 =="
"$PY" - "$MEAN" "$STD" <<'PYEOF'
import glob, onnx, numpy as np, sys
mean, std = float(sys.argv[1]), float(sys.argv[2])
for f in sorted(glob.glob('cand/steps-3-melfix-ep*.onnx')):
    vals = np.concatenate([onnx.numpy_helper.to_array(i).ravel().astype(float)
                           for i in onnx.load(f).graph.initializer
                           if onnx.numpy_helper.to_array(i).dtype.kind == 'f'])
    print(f, '| V1 对 (-2.9522,5.4468):',
          int(np.isclose(vals, -2.9522, atol=1e-4).sum()), int(np.isclose(vals, 5.4468, atol=1e-4).sum()),
          f'| 训练实际用的一对 ({mean:.4f},{std:.4f}):',
          int(np.isclose(vals, mean, atol=1e-4).sum()), int(np.isclose(vals, std, atol=1e-4).sum()))
PYEOF

echo "== 5) 打包（回 Mac 跑 ruler 对 25.8% 基线）=="
tar czf /root/onnx_melfix.tgz cand/steps-3-melfix-ep*.onnx 2>/dev/null
ls -lh /root/onnx_melfix.tgz
