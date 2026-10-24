#!/usr/bin/env python3
"""SenseVoice 回读公共件：加载识别器 + 文本归一比较。"""
import re

import config

_STRIP = re.compile(r"[\s，。、；;,.!！?？:：'\"“”()（）\-]")


def norm(s):
    return _STRIP.sub("", s.lower())


def load_recognizer(threads=4):
    import sherpa_onnx as so
    return so.OfflineRecognizer.from_sense_voice(
        model=config.sv_file("model.int8.onnx"),
        tokens=config.sv_file("tokens.txt"),
        num_threads=threads, language="", use_itn=False, debug=False)


def transcribe(rec, samples, sample_rate=16000):
    s = rec.create_stream()
    s.accept_waveform(sample_rate, list(map(float, samples)))
    rec.decode_stream(s)
    return s.result.text.strip()
