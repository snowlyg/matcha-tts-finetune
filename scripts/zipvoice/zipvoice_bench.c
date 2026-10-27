// zipvoice_bench.c — 端侧 ZipVoice 基准（sherpa-onnx v2.0.0 C API）
// 一个进程内加载一次模型，逐句生成，打印 elapsed/audio/rtf，并把 wav 落盘供人耳复听
//
// 为什么留这个文件：ZipVoice 是"中英混读换基座"的候选，在真实设备上量到 RTF 33~47（中位 ~35，
// 一条 2 秒播报要 68~74 秒），同机同口径的 Matcha V1 是 1.20 —— 差约 29 倍，直接判死。
// 判死要留能量出这把尺子的东西，不是只留一句结论。复现步骤和两个版本边界（现装的 1.12.18
// 加载不了这个 int8 包，要 ≥v1.12.20；v2.0.0 又把 espeak 移掉了，英文词典得单独喂）见同目录 README.md。
//
// 用法：./zipvoice_bench <model_dir> <vocos_24k> <lexicon-en-us.txt> <phrases> [steps] [threads] [out_dir]
// steps 是流匹配采样步数、耗时随它近似线性走，所以报数必须同时写步数。
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sherpa-onnx/c-api/c-api.h"

static double NowSec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int32_t QuietCallback(const float *samples, int32_t num_samples,
                             float progress, void *arg) {
  (void)samples; (void)num_samples; (void)progress; (void)arg;
  return 1;
}

static void PrintPeakRss(void) {
  FILE *fp = fopen("/proc/self/status", "r");
  if (!fp) return;
  char line[256];
  long hwm = 0, vmpeak = 0;
  while (fgets(line, sizeof(line), fp)) {
    if (strncmp(line, "VmHWM:", 6) == 0) sscanf(line + 6, "%ld", &hwm);
    if (strncmp(line, "VmPeak:", 7) == 0) sscanf(line + 7, "%ld", &vmpeak);
  }
  fclose(fp);
  fprintf(stderr, "PEAK_RSS_kb=%ld VmPeak_kb=%ld\n", hwm, vmpeak);
}

int32_t main(int32_t argc, char *argv[]) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s <model_dir> <vocos> <lexicon_en_us> <phrases_file> "
            "[steps=4] [threads=4] [out_dir=out]\n", argv[0]);
    return 1;
  }
  const char *model_dir = argv[1];
  const char *vocos = argv[2];
  const char *lex_en = argv[3];
  const char *phrases_file = argv[4];
  int32_t steps = argc > 5 ? atoi(argv[5]) : 4;
  int32_t threads = argc > 6 ? atoi(argv[6]) : 4;
  const char *out_dir = argc > 7 ? argv[7] : "out";

  char path[1024];
  char tokens[1024], enc[1024], dec[1024], lex[1200];
  snprintf(tokens, sizeof(tokens), "%s/tokens.txt", model_dir);
  snprintf(enc, sizeof(enc), "%s/encoder.int8.onnx", model_dir);
  snprintf(dec, sizeof(dec), "%s/decoder.int8.onnx", model_dir);
  snprintf(lex, sizeof(lex), "%s/lexicon.txt,%s", model_dir, lex_en);

  // 参考音频（5-10 秒零样本克隆的音色来源）
  char ref_wav[1024];
  snprintf(ref_wav, sizeof(ref_wav), "%s/test_wavs/news-female-2.wav", model_dir);
  const char *ref_text =
      "本台消息, 中共中央国务院, 近日印发关于构建数据基础制度, 更好发挥数据要素作用的意见.";

  SherpaOnnxOfflineTtsConfig config;
  memset(&config, 0, sizeof(config));
  config.model.zipvoice.tokens = tokens;
  config.model.zipvoice.encoder = enc;
  config.model.zipvoice.decoder = dec;
  config.model.zipvoice.vocoder = vocos;
  config.model.zipvoice.lexicon = lex;
  config.model.zipvoice.data_dir = NULL;  // deprecated in v2.0.0
  // C API 结构体没有默认值：这几个必须显式赋，否则 Validate 判 feat_scale<=0
  config.model.zipvoice.feat_scale = 0.1f;
  config.model.zipvoice.t_shift = 0.5f;
  config.model.zipvoice.target_rms = 0.1f;
  config.model.zipvoice.guidance_scale = 1.0f;
  config.model.num_threads = threads;
  config.model.debug = getenv("ZV_DEBUG") ? 1 : 0;
  config.model.provider = "cpu";
  config.silence_scale = 0.2f;
  config.max_num_sentences = 1;

  double t0 = NowSec();
  const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
  fprintf(stderr, "create_ms=%d threads=%d steps=%d\n",
          (int)((NowSec() - t0) * 1000), threads, steps);
  if (!tts) {
    fprintf(stderr, "FATAL: SherpaOnnxCreateOfflineTts returned null\n");
    return 2;
  }
  PrintPeakRss();

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(ref_wav);
  if (!wave) {
    fprintf(stderr, "FATAL: failed to read reference %s\n", ref_wav);
    SherpaOnnxDestroyOfflineTts(tts);
    return 3;
  }
  fprintf(stderr, "reference=%s sr=%d samples=%d dur=%.2fs\n", ref_wav,
          wave->sample_rate, wave->num_samples,
          (float)wave->num_samples / wave->sample_rate);

  FILE *fp = fopen(phrases_file, "r");
  if (!fp) {
    fprintf(stderr, "FATAL: cannot open %s\n", phrases_file);
    return 4;
  }
  char text[512];
  int32_t idx = 0;
  while (fgets(text, sizeof(text), fp)) {
    size_t n = strlen(text);
    while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r')) text[--n] = 0;
    if (n == 0) continue;
    idx++;

    SherpaOnnxGenerationConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.speed = 1.0f;
    cfg.num_steps = steps;
    cfg.reference_audio = wave->samples;
    cfg.reference_audio_len = wave->num_samples;
    cfg.reference_sample_rate = wave->sample_rate;
    cfg.reference_text = ref_text;

    double g0 = NowSec();
    const SherpaOnnxGeneratedAudio *audio =
        SherpaOnnxOfflineTtsGenerateWithConfig(tts, text, &cfg, QuietCallback, NULL);
    double el = NowSec() - g0;

    if (!audio || audio->n <= 0) {
      fprintf(stderr, "CASE=%d | EMPTY_AUDIO | text=%s\n", idx, text);
      if (audio) SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
      continue;
    }
    float dur = (float)audio->n / audio->sample_rate;
    snprintf(path, sizeof(path), "%s/zv_%02d_s%d.wav", out_dir, idx, steps);
    SherpaOnnxWriteWave(audio->samples, audio->n, audio->sample_rate, path);
    fprintf(stderr,
            "CASE=%d | elapsed=%.3f s dur=%.3f s rtf=%.3f | sr=%d | text=%s\n",
            idx, el, dur, el / dur, audio->sample_rate, text);
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
  }
  fclose(fp);
  SherpaOnnxFreeWave(wave);
  fprintf(stderr, "CASES=%d\n", idx);
  PrintPeakRss();
  SherpaOnnxDestroyOfflineTts(tts);
  fprintf(stderr, "BENCH_DONE\n");
  return 0;
}
