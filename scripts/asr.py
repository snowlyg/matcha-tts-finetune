#!/usr/bin/env python3
"""SenseVoice 回读公共件：加载识别器 + 文本归一比较。"""
import re

import config

_STRIP = re.compile(r"[\s，。、；;,.!！?？:：'\"“”()（）\-]")


def norm(s):
    return _STRIP.sub("", s.lower())


def py(s):
    """音位串：同音字在 ASR 里是随机选的，字符口径会把「输液→书叶」记成真错。

    尺子同时报"字"与"音"两个口径：字口径包含同音选字假错（偏严、可与人耳对齐），
    音口径只看读音是否一致（用来判断错率涨幅里有多少是真损伤）。
    """
    from pypinyin import lazy_pinyin, Style
    return tuple(lazy_pinyin(norm(s), style=Style.TONE3,
                             neutral_tone_with_five=True, errors="default"))


def same_sound(heard, exp):
    return py(heard) == py(exp)


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
