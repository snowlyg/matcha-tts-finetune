#!/usr/bin/env python3
"""PC 复现：用作者 V1 的 ONNX（声学+vocos）+ 本项目 g2p 合成话术。

输出 wav + token 数 + RTF，供与端侧 sherpa 输出对照。不改模型，只验证链路。
喂入用规整话术：字母拆 I;C;U;、数字转汉字。
"""
import argparse
import os
import time

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf

import config
import g2p

DEFAULT_PHRASES = [
    "体温三十六点五度",
    "二十一床正在呼叫",
    "为您自动接听",
    "A;一零床正在呼叫",
    "I;C;U;三床请到护士站",
]


class OnnxModel:
    def __init__(self, filename, threads):
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.inter_op_num_threads = 1
        self.m = ort.InferenceSession(filename, sess_options=o,
                                      providers=["CPUExecutionProvider"])
        self.sample_rate = int(self.m.get_modelmeta().custom_metadata_map["sample_rate"])

    def __call__(self, x):
        x = np.asarray(x, dtype=np.int64)[None, :]
        return self.m.run([self.m.get_outputs()[0].name], {
            self.m.get_inputs()[0].name: x,
            self.m.get_inputs()[1].name: np.array([x.shape[1]], dtype=np.int64),
            self.m.get_inputs()[2].name: np.array([1.0], dtype=np.float32),
            self.m.get_inputs()[3].name: np.array([1.0], dtype=np.float32),
        })[0]


class OnnxVocos:
    def __init__(self, filename):
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        self.m = ort.InferenceSession(filename, sess_options=o,
                                      providers=["CPUExecutionProvider"])

    def __call__(self, x):
        return self.m.run([n.name for n in self.m.get_outputs()],
                          {self.m.get_inputs()[0].name: x})


def istft_audio(mag, x, y):
    stft = knf.StftResult(
        real=(mag * x)[0].transpose().reshape(-1).tolist(),
        imag=(mag * y)[0].transpose().reshape(-1).tolist(),
        num_frames=mag.shape[2])
    cfg = knf.StftConfig(n_fft=1024, hop_length=256, win_length=1024,
                         window_type="hann", center=True, pad_mode="reflect",
                         normalized=False)
    return np.array(knf.IStft(cfg)(stft))


def safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:40]


def build(threads=4):
    _, tok2id = g2p.load_vocab()
    am = OnnxModel(config.acoustic_path(), threads)
    voc = OnnxVocos(config.vocoder_path())
    return am, voc, tok2id


def synth(am, voc, tok2id, text, bos=None):
    ids = g2p.text_to_ids(text, tok2id, bos=bos)
    mag, x, y = voc(am(ids))
    return ids, istft_audio(mag, x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(config.OUT, "repro"))
    ap.add_argument("--bos", type=int, default=None)
    ap.add_argument("--text-file", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    am, voc, tok2id = build(args.threads)
    phrases = ([l.strip() for l in open(args.text_file, encoding="utf-8") if l.strip()]
               if args.text_file else DEFAULT_PHRASES)
    for text in phrases:
        t0 = time.time()
        ids, audio = synth(am, voc, tok2id, text, bos=args.bos)
        dt = time.time() - t0
        dur = len(audio) / am.sample_rate
        name = safe(text) + ".wav"
        sf.write(os.path.join(args.out, name), audio, am.sample_rate, "PCM_16")
        print(f"{text!r:30s} tokens={len(ids):3d} audio={dur:5.2f}s "
              f"rtf={dt/dur:5.2f} -> {name}", flush=True)


if __name__ == "__main__":
    main()
