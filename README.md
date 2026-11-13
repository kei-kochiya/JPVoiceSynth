# JPVoiceSynth — Multi-Speaker Japanese Neural TTS (`GPT-SoVITS v2ProPlus`)

End-to-end audio curation, DSP cleaning, two-stage fine-tuning (`GPT` + `SoVITS` with `ERes2NetV2` speaker verification), GPU-aware batch scheduling, and sub-real-time Japanese TTS inference built on [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (`v2ProPlus`) for **NVIDIA RTX 5070 (Blackwell `sm_120`, CUDA 12.8)**.

> **Copyright & Asset Notice:** This repository contains **training, preprocessing, scheduling, and inference code only**. All source audio, game transcripts, reference clips, pretrained models, and fine-tuned weights are excluded via `.gitignore` and are not redistributed here. To use this pipeline, supply your own legally obtained/consented audio dataset.

---

## Architecture & Pipeline

1. **Metadata Filtering & Corpus Curation** (`scripts/collect_ichika_metadata.py`, `scripts/collect_and_process_leoneed.py`)
   - Scans story manifests, isolates solo single-speaker lines, and filters out multi-speaker clips, alternate age/mob IDs, hesitation-only fragments, and short (`<12` char) utterances.
2. **Audio DSP Cleaning & Anti-Breath Preprocessing** (`scripts/preprocess_ichika_audio.py`, `scripts/collect_and_process_leoneed.py`)
   - Decodes audio to `32 kHz` mono PCM with an `80 Hz` 2-pole Butterworth high-pass filter (`ffmpeg`).
   - Trims edge silence with **Silero VAD** (`threshold=0.45`, `speech_pad_ms=65`) and enforces strict duration (`2.0–12.0s`), voiced-ratio (`>= 0.55`), speech-RMS (`>= 0.012`), and non-clipping filters.
   - Applies gentle stationary spectral denoising (`noisereduce`), `-18 dB` soft-gating on inter-word gaps `>160 ms`, and `35 ms` cosine edge fades.
   - Curated **`25,121` clean utterances (`~36.2 hours`)** across 4 speakers (`9.53h`, `11.19h`, `8.56h`, `6.93h`).
3. **Two-Stage `v2ProPlus` Training & GPU Queue Scheduler** (`scripts/run_gptsovits_pipeline.py`, `scripts/run_leoneed_queue.py`)
   - Stage 1: `pyopenjtalk` + `RoBERTa` text/phoneme features, `CNHubert` `25 Hz` semantic tokens, and `ERes2NetV2` speaker embeddings.
   - Stage 2: Mixed-precision (`FP16` / `16-mixed`) fine-tuning of SoVITS (`s2`) and GPT (`s1`) with automatic `nvidia-smi` GPU contention detection (`wait_for_gpu_free`), post-run cache cleanup, and verification synthesis.
4. **Chunked Long-Form Japanese Inference** (`scripts/synthesize.py`, `scripts/eval_4h_checkpoints.py`)
   - Splits multi-sentence Japanese text on terminal punctuation (`。`, `！`, `？`) with configurable pauses (`--pause_second 0.45`) to avoid long-horizon autoregressive prosody decay.

---

## How to Use

### 1. Clone & Install Dependencies (Windows / CUDA 12.8)
Requires **Git**, **FFmpeg** on `PATH`, **Python 3.11**, and **uv** (or `pip`). Tested on Windows with RTX 5070 (`sm_120`, CUDA 12.8) without requiring MSVC build tools:

```powershell
git clone https://github.com/kei-kochiya/JPVoiceSynth.git
cd JPVoiceSynth

# Clone upstream GPT-SoVITS inside ./GPT-SoVITS
git clone https://github.com/RVC-Boss/GPT-SoVITS.git GPT-SoVITS

# Create Python 3.11 venv and install PyTorch cu128 + dependencies
uv venv GPT-SoVITS\.venv --python 3.11
uv pip install --python GPT-SoVITS\.venv\Scripts\python.exe torch==2.7.1+cu128 torchaudio==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python GPT-SoVITS\.venv\Scripts\python.exe -c setup\torch-constraints.txt -r setup\requirements-uv-cu128.txt
```

Download the `GPT-SoVITS v2ProPlus` pretrained models (`s2Gv2ProPlus.pth`, `s2Dv2ProPlus.pth`, `s1v3.ckpt`, `chinese-hubert-base`, `chinese-roberta-wwm-ext-large`, and `sv/pretrained_eres2netv2w24s4ep4.ckpt`) into `GPT-SoVITS/GPT_SoVITS/pretrained_models/` per upstream instructions.

### 2. Supply Your Own Voice Dataset
You do **not** need the game dataset scripts to train on your own Japanese speaker(s):
1. Prepare clean mono `32 kHz` WAV clips (`2–12s` each, dry mic, minimal background noise). If starting from raw noisy clips, adapt `scripts/preprocess_ichika_audio.py` to trim silence with Silero VAD, high-pass at `80 Hz`, and filter clipped/quiet segments.
2. Create a UTF-8 dataset manifest `data/my_voice.list` using GPT-SoVITS's 4-column pipe format (`audio_path|speaker_name|language|transcript`):
   ```text
   data/my_wavs/clip_0001.wav|my_speaker|ja|放課後の音楽室って、すごく落ち着くんだよね。
   data/my_wavs/clip_0002.wav|my_speaker|ja|今日もみんなでスタジオに集まって練習しよう！
   ```
   *(Make sure every audio file is `32 kHz` mono WAV and its transcript accurately matches the spoken Japanese line.)*

### 3. Train Stage 1 + SoVITS (`s2`) + GPT (`s1`)
Run the automated training runner on your `.list` file:
```powershell
GPT-SoVITS\.venv\Scripts\python.exe -u scripts\run_gptsovits_pipeline.py `
  --exp_name my_voice `
  --list_path data\my_voice.list `
  --s2_batch 6 --s2_epochs 6 --s2_save_every 3 `
  --s1_batch 6 --s1_epochs 10 --s1_save_every 5 `
  --n_parts 2
```
- **Outputs:** Exported weights land in `GPT-SoVITS/SoVITS_weights_v2ProPlus/my_voice_e6_*.pth` and `GPT-SoVITS/GPT_weights_v2ProPlus/my_voice-e10.ckpt`, plus a timing JSON at `data/benchmark_my_voice.json`.
- **Scaling tip:** For smaller datasets (`1–3 hours`), increase epochs (`--s2_epochs 10 --s1_epochs 15`); for large corpora (`7–11 hours`), `6` SoVITS epochs and `10` GPT epochs give plenty of steps without overfitting.
- **Multi-speaker queue:** Edit `JOBS` in `scripts/run_leoneed_queue.py` to train multiple speakers sequentially with automatic `nvidia-smi` GPU availability waiting and post-run cache cleanup.

### 4. Synthesize Speech
Pick one super clean, dry, neutral `3.5–6s` WAV clip (avoid whispery/breathy takes unless you want breathy output) as your reference prompt:
```powershell
GPT-SoVITS\.venv\Scripts\python.exe scripts\synthesize.py `
  --gpt_model GPT-SoVITS\GPT_weights_v2ProPlus\my_voice-e10.ckpt `
  --sovits_model GPT-SoVITS\SoVITS_weights_v2ProPlus\my_voice_e6_sXXXX.pth `
  --ref_audio references\my_ref.wav `
  --ref_text "Reference clip exact Japanese transcript" `
  --text "今日はいい天気ですね。放課後に練習を始めましょう！" `
  --out outputs\test.wav `
  --cut sentence --pause_second 0.45 --top_p 0.6 --temperature 0.6
```

---

## Benchmarks (RTX 5070 12 GB, CUDA 12.8)

- **4-hour single-speaker benchmark (`2,691` lines, `SoVITS` 8 ep + `GPT` 12 ep, `bs=6`):** **`52.71 min`** total (`~4.8 min` Stage 1 prep, `32.13 min` SoVITS, `15.80 min` GPT).
- **Full multi-speaker runs (`SoVITS` 6 ep + `GPT` 10 ep, `bs=6`):**
  - `6,341` lines (`9.53h`): `~84.7 min` active compute (`~7.0 min` prep, `57.65 min` SoVITS, `19.75 min` GPT)
  - `7,680` lines (`11.19h`): **`107.47 min`** (`6.79 min` prep, `74.33 min` SoVITS, `26.14 min` GPT)
  - `5,829` lines (`8.56h`): **`81.68 min`** (`6.59 min` prep, `55.02 min` SoVITS, `19.85 min` GPT)
  - `5,271` lines (`6.93h`): **`73.48 min`** (`5.90 min` prep, `49.13 min` SoVITS, `18.24 min` GPT)
- **Inference speed (warmed-up `v2ProPlus` FP16 + CUDA Graph):** **`0.562–0.614` RTF** (`12.73s` GPU time for `21.55s` of `32 kHz` audio).

---

## Repository Structure

```text
scripts/
  collect_ichika_metadata.py      # Metadata indexing & solo-line filtering
  download_ichika_mp3s.py         # Parallel audio downloader
  preprocess_ichika_audio.py      # Silero VAD, HPF, denoise, & soft pause gating
  collect_and_process_leoneed.py  # Multi-speaker indexing, download, & DSP pipeline
  run_gptsovits_pipeline.py       # Stage 1 feature prep + s2/s1 training runner
  run_leoneed_queue.py            # Sequential queue with nvidia-smi GPU wait & cleanup
  synthesize.py                   # Sentence-chunked Japanese CLI inference
  eval_4h_checkpoints.py          # Long-sequence checkpoint evaluation harness
setup/
  torch-constraints.txt           # PyTorch 2.7.1+cu128 pins for Blackwell sm_120
  requirements-uv-cu128.txt       # Reproducible Windows dependencies (no MSVC needed)
docs/
  tts_ai_and_nlp_guide.md         # Architecture & Generative AI / NLP notes
```
