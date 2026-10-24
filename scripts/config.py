"""集中路径配置：所有脚本从这里取路径，便于换机器/云环境。

默认放在仓库内 models/ 与 out/（均 .gitignore，不入库）。
换机器时用环境变量覆盖：TTS_VOCAB / TTS_MODEL_DIR / TTS_SV_DIR / TTS_OUT。
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(key, default):
    return os.environ.get(key, default)


# 作者 V1 中英混读 matcha 模型（从 ModelScope 下载，不入库）
MODEL_DIR = _env("TTS_MODEL_DIR", os.path.join(REPO, "models", "zh-en"))
ACOUSTIC = _env("TTS_ACOUSTIC", "model-steps-3.onnx")
VOCODER = _env("TTS_VOCODER", "vocos-16khz-univ.onnx")

# 词表（= 设备端 tokens.txt 的 token 列，2190 条）
VOCAB = _env("TTS_VOCAB", os.path.join(REPO, "models", "vocab_tts.txt"))

# SenseVoice 回读模型（尺子/筛选打分用，从 sherpa-onnx release 下载，不入库）
SV_DIR = _env("TTS_SV_DIR", os.path.join(REPO, "models", "sense-voice"))

OUT = _env("TTS_OUT", os.path.join(REPO, "out"))


def acoustic_path():
    return os.path.join(MODEL_DIR, ACOUSTIC)


def vocoder_path():
    return os.path.join(MODEL_DIR, VOCODER)


def sv_file(name):
    return os.path.join(SV_DIR, name)
