#!/usr/bin/env python3
"""中英混读 matcha 的文本->token 前端（G2P），用于 PC 复现与尺子。

复原自作者 V1 测试脚本 test_onnx_20251203_zh_en.py，并对齐 icefall 训练前端
（jieba + lazy_pinyin(tone_sandhi=True) + piper_phonemize）。注意：训练侧应以
icefall 的 local/convert_text_to_tokens.py 为准；本模块只做 PC 端复现/打分。

英文后端优先 piper_phonemize；不可用时退回系统 espeak-ng CLI（近似）。
"""
import argparse
import os
import re
import subprocess

from pypinyin import lazy_pinyin, Style

import config

ZWJ = "‍"
_PUNCT = {"，": ",", "。": ".", "！": "!", "？": "?"}


def load_vocab(path=None):
    path = path or config.VOCAB
    with open(path, encoding="utf-8") as f:
        vocab = [x.strip() for x in f.readlines()]
    return vocab, {p: i + 1 for i, p in enumerate(vocab)}


def chinese_to_pinyin(text):
    return lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True,
                       tone_sandhi=True)


def convert_to_gruut_en_us_strict(ipa_input):
    text = "".join(ipa_input) if isinstance(ipa_input, list) else ipa_input
    text = text.replace(ZWJ, "")
    for pat, rep in [
        ("ɝ", "ɜɹ"), ("ɚ", "əɹ"),
        ("eɪ", "A"), ("aɪ", "I"), ("ɔɪ", "Y"), ("oʊ", "O"), ("əʊ", "O"), ("aʊ", "W"),
        ("tʃ", "ʧ"), ("dʒ", ""),
        ("ː", ""), ("g", "ɡ"), ("r", "ɹ"), ("e", "ɛ"),
    ]:
        text = text.replace(pat, rep)
    return text


def phonemize_en(word):
    try:
        from piper_phonemize import phonemize_espeak
        return phonemize_espeak(word, "en-us")[0]
    except Exception:
        out = subprocess.run(["espeak-ng", "-v", "en-us", "-q", "--ipa=3", word],
                             capture_output=True, text=True, check=True).stdout
        return out.strip()


def text_to_ids(text, tok2id, bos=None):
    ids = [] if bos is None else [bos]
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if "一" <= c <= "鿿":
            j = i
            while j < n and "一" <= text[j] <= "鿿":
                j += 1
            ids.extend(tok2id.get(p, 1) for p in chinese_to_pinyin(text[i:j]))
            i = j
        elif c.isascii() and c.isalpha():
            j = i
            while j < n and text[j].isascii() and text[j].isalpha():
                j += 1
            ids.extend(tok2id.get(ch, 1)
                       for ch in convert_to_gruut_en_us_strict(phonemize_en(text[i:j])))
            i = j
        else:
            ids.append(tok2id.get(_PUNCT.get(c, c), 1))
            i += 1
    return ids


# ---- 自检：复现作者 test_sherpa_onnx.py 里的硬编码 id 序列 ----
_SELF_TEXT = (
    "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，"
    "思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，"
    "感受着生命的奇迹与温柔。某某银行的副行长和一些行政领导表示，他们去过长江和长白山，"
    "经济不断增长。二零二四年十二月三十一号，拨打幺幺零或者幺八九二零二四零五幺幺。"
    "十两万三千四百五十六块钱。"
)
_SELF_EXPECTED = [
    414, 1944, 1169, 799, 1012, 4, 1886, 661, 453, 453, 4, 154, 1697, 2065, 1826,
    561, 582, 1129, 4, 1843, 2004, 819, 1124, 2086, 598, 1629, 2065, 1622, 661,
    420, 1022, 2113, 4, 1671, 1229, 1512, 992, 1946, 414, 1934, 4, 1119, 819,
    1512, 749, 839, 2053, 856, 4, 1843, 1978, 2139, 1467, 1502, 1827, 1949, 1748,
    4, 292, 819, 2004, 2069, 1359, 1252, 819, 420, 1108, 984, 2081, 2086, 4, 598,
    1629, 2065, 1616, 1149, 420, 1397, 784, 1978, 1831, 1507, 5, 1163, 1163, 1952,
    697, 420, 584, 697, 2058, 707, 1946, 1876, 1887, 2079, 1018, 418, 193, 1624,
    4, 1711, 1110, 1449, 679, 277, 796, 707, 277, 147, 1591, 4, 816, 784, 217,
    494, 2036, 2058, 5, 534, 1017, 534, 1674, 1227, 1622, 534, 1989, 1551, 1622,
    1946, 704, 4, 211, 398, 1936, 1936, 1017, 779, 2068, 1936, 141, 828, 534,
    1017, 534, 1674, 1017, 1848, 1936, 1936, 5, 1622, 998, 1819, 1551, 1406, 1674,
    148, 1848, 1622, 1024, 914, 1407, 5,
]
_KNOWN = {(411, 414), (1946, 1949), (1949, 1946), (219, 217)}  # BOS + 个别多音字(未做 jieba 分词)


def selftest(vocab_path=None):
    _, tok2id = load_vocab(vocab_path)
    got = text_to_ids(_SELF_TEXT, tok2id)
    unknown = [(i, a, b) for i, (a, b) in enumerate(zip(got, _SELF_EXPECTED))
               if a != b and (a, b) not in _KNOWN]
    print(f"len got={len(got)} expected={len(_SELF_EXPECTED)} "
          f"unknown_diffs={len(unknown)}")
    for i, a, b in unknown[:10]:
        print(f"  at {i}: got={a} expected={b}")
    return len(unknown)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--text", default="")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() == 0 else 1)
    if args.text:
        _, tok2id = load_vocab()
        print(text_to_ids(args.text, tok2id))
