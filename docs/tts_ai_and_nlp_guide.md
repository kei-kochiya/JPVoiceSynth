# How Modern Neural Voice Synthesis Works (AI, NLP, and Your Project)

## 1. Short Answer: Is This AI or NLP?

**Yes — it is both.** More precisely, modern Text-to-Speech (**Neural TTS**) sits right at the intersection of **Artificial Intelligence (Deep Learning)**, **Natural Language Processing (NLP)**, and **Digital Signal Processing (DSP)**:

```text
AI (Artificial Intelligence)
 └── Deep Learning / Generative AI
      ├── NLP (Text -> Linguistics, Phonemes, Context Embeddings)
      ├── Audio Signal Processing (Filtering, VAD, Denoising, Mel Spectrograms)
      └── Speech Generative Models (GPT Autoregressive Tokens + SoVITS Acoustic Decoder)
```

- **Why it counts as NLP:** Raw Japanese text (`"放課後の音楽室"`) cannot be turned into sound directly because kanji readings, pitch contours, and phrasing depend on grammar and meaning. The front end runs morphological analysis, grapheme-to-phoneme (**G2P**) conversion, and a **RoBERTa** language model to extract linguistic meaning.
- **Why it counts as Generative Audio AI:** Knowing *what words to say* isn't sound. The second half of the pipeline predicts discrete acoustic-semantic tokens (like a language model, except its vocabulary is tiny chunks of speech instead of text words) and synthesizes a `32,000 Hz` waveform with a conditional VAE + GAN vocoder.

---

## 2. The End-to-End Architecture of Your Pipeline (`GPT-SoVITS v2ProPlus`)

Instead of one giant black box going straight from Japanese text to WAV audio, `GPT-SoVITS` splits the problem into **three cleanly separated stages**:

```text
[Raw Game Audio + Scripts]
        │
        ▼  (0. DSP & Curation: preprocess_ichika_audio.py / collect_and_process_leoneed.py)
Clean 32 kHz Mono WAVs (2–12s) + Japanese Text
        │
        ├──► 1a. NLP Front-End (pyopenjtalk + RoBERTa) ──► Phonemes + Linguistic Vectors
        ├──► 1b. Speech Encoders (CNHubert + ERes2NetV2) ─► Acoustic Features + Speaker Print
        └──► 1c. VQ Quantizer (25 Hz) ────────────────────► Discrete Semantic Tokens (0..1023)
                                                                  │
        ┌─────────────────────────────────────────────────────────┘
        ▼
[Stage 1: GPT ("Text2Semantic", s1)]
   Input:  Phonemes + BERT embeddings + Reference audio semantic prompt
   Output: Sequence of 25 Hz semantic tokens (prosody, rhythm, intonation, pauses)
        │
        ▼
[Stage 2: SoVITS ("Semantic2Wave", s2)]
   Input:  Generated semantic tokens + Phonemes + Reference timbre / ERes2NetV2 speaker vector
   Output: 32,000 Hz audio waveform (timbre, vocal clarity, formants, realism)
```

---

## 3. Breaking Down Each Layer

### Stage 0: Audio Curation & Signal Processing (Why Clean Data Matters Most)
Deep learning models imitate *everything* in their training recordings — including room hum, clipping distortion, and heavy breaths. Before training anything, our preprocessing pipeline does four DSP operations:
1. **80 Hz High-Pass Filter (`highpass=f=80:poles=2`):** Cuts sub-bass microphone thumps and desk rumble that carry zero human speech information.
2. **Silero VAD (`Voice Activity Detection`):** A lightweight neural network scans the clip at `16 kHz` to separate real speech from dead air, trimming edge silence (`speech_pad_ms=65`) and rejecting clips that are clipped, too quiet, too short (`<2.0s`), too long (`>12.0s`), or mostly silence (`voiced_ratio < 0.55`).
3. **Gentle Stationary Spectral Denoising (`noisereduce`):** Estimates the background floor from non-speech frames inside the raw clip and reduces steady hiss mildly (`prop_decrease=0.30–0.42`) without creating metallic "underwater" artifacts.
4. **Soft Pause Gating & Cosine Edge Fades:** Gently attenuates (`-18 dB`) long non-speech gaps (`>160ms`) and fades edges (`35ms`) so concatenated training batches have zero clicks.

Across all four characters (`ichika`, `saki`, `honami`, `shiho`), this curated **25,121 clean utterances (~36.2 hours)** from tens of thousands of raw game voice lines.

---

### Stage 1 (`1a`, `1b`, `1c`): Turning Words and Sound Into Numbers

#### `1a` — NLP Text Front-End (`1-get-text.py`)
- **Normalization & G2P (`pyopenjtalk`):** Converts mixed kanji/hiragana/katakana (`"今日"`, `"私達"`, `"セットリスト"`) into unambiguous **phoneme IDs** (sounds like `k y o u`, `w a t a sh i t a ch i`). In Japanese, the exact same kanji has multiple readings (`今日` = *kyou* vs *konnichi*), so syntactic parsing is essential.
- **Contextual Language Embeddings (`chinese-roberta-wwm-ext-large`):** Extracts dense transformer representations encoding word boundaries and context so prosody flows naturally across phrases rather than reading word-by-word like a robot.

#### `1b` & `1c` — Self-Supervised Speech Tokens & Speaker Identity (`2-get-*`, `3-get-semantic.py`)
- **CNHubert (`chinese-hubert-base`):** A self-supervised speech representation model (similar in spirit to Wav2Vec2 / HuBERT). It reads `16 kHz` audio and captures phonetic content and pronunciation nuances without needing human frame-by-frame labels.
- **Vector Quantization (`25 Hz` Semantic Tokens):** A frozen quantizer (`freeze_quantizer: true` in `s2v2ProPlus.json`) compresses continuous `CNHubert` features into a discrete vocabulary of **`1,024` audio-semantic tokens** (`vocab_size: 1025` including `EOS: 1024`), at **25 tokens per second** (`40ms` per token).
- **Speaker Verification Encoder (`ERes2NetV2`):** Reads the `16 kHz` voice sample and produces a fixed-size speaker-identity embedding ("voiceprint"), which `v2ProPlus` uses alongside reference spectrograms to lock onto the speaker's vocal tract timbre.

---

### Stage 2: Fine-Tuning the Two Neural Models

#### Model A — GPT (`s1_train.py`, `s1longer-v2.yaml`): The "Director"
- **Architecture:** A 24-layer, 16-head, `512`-hidden-dimension **autoregressive Transformer decoder** (`phoneme_vocab_size: 732`, `vocab_size: 1025`).
- **Job:** Predicts the *next* `40ms` speech token one step at a time, conditioned on the input phonemes, text embeddings, and the reference prompt's semantic tokens.
- **What it controls:** Speaking rhythm, syllable duration, pitch rise/fall (questions vs statements), hesitations, and where to pause.
- **Understanding `top_3_acc`:** During training (`e.g. 0.48 -> 0.65` on `ichika_4h`), `top_3_acc` measures how often the true next `40ms` token from the real actor is inside the model's top 3 guesses out of `1,025` choices. Because natural human speech has many equally valid ways to time a syllable or pause, `100%` is neither expected nor desirable (`0.60–0.70` usually reflects strong prosody learning without memorizing).
- **Inference Controls (`top_k`, `top_p`, `temperature`):**
  - Lower `temperature` / `top_p` (e.g. `0.6`) -> steadier, cleaner, less erratic delivery.
  - Higher `temperature` / `top_p` (e.g. `0.8–1.0`) -> more varied emotion, slightly higher chance of weird pacing.

#### Model B — SoVITS (`s2_train.py`, `s2v2ProPlus.json`): The "Voice Actor / Renderer"
- **Architecture:** A conditional **Variational Autoencoder (VAE)** with normalizing flows, coupled with an adversarial **Multi-Scale / Multi-Period Discriminator** (`s2D`) and a HiFi-GAN-style neural vocoder (`s2G`).
  - Audio spec: `32,000 Hz`, `hop_length = 640`, `128` mel channels.
  - Upsampling chain: `[10, 8, 2, 2, 2]` -> `10 × 8 × 2 × 2 × 2 = 640`, turning `50 Hz` acoustic frames cleanly into `32,000` raw audio samples per second.
- **Job:** Takes the semantic tokens generated by GPT, combines them with the phoneme sequence and the `ERes2NetV2` / reference acoustic timbre (`gin_channels: 1024`), and generates crisp `32 kHz` audio waveforms.
- **Why there are two checkpoints during training (`s2G` and `s2D`):**
  - `s2G` (Generator) tries to synthesize realistic audio.
  - `s2D` (Discriminator) tries to spot synthetic audio vs real ground-truth recordings.
  - Training them against each other forces `s2G` to preserve crisp consonants, breath texture, and harmonic brightness instead of blurry averaged spectrograms. At inference time, you only load `s2G` (exported as `SoVITS_weights_v2ProPlus/*.pth`).

---

## 4. Why Fine-Tuning Beats Pure Zero-Shot Cloning

- **Zero-shot cloning** tries to infer a speaker's entire voice identity, accent habits, and emotional range from a single `3–5s` reference clip on the fly. It gets the rough timbre, but often drifts on long sentences or misses character-specific pitch drops and sentence endings.
- **Fine-tuning (`6–10` epochs over `7–11` hours)** adapts the weights of **both** GPT (learning Ichika/Saki/Honami/Shiho's real rhythm, cadence, and Japanese phrasing) and SoVITS (learning their exact vocal harmonics across whispers, calm narration, and energetic shouting).
- Even *after* fine-tuning, the **`3–10s` reference clip** still acts as a style steer at inference time — which is why switching Ichika's reference from a soft emotional whisper (`0.084` RMS, `23.8%` breath frames) to a crisp conversational clip (`0.177` RMS, `10.7%` breath frames) immediately halved excess breathiness and doubled vocal projection!

---

## 5. Key Engineering Lessons Observed in This Project

1. **Autoregressive Horizon & Sentence Splitting:**
   - The training clips are `2–12s`. When GPT-SoVITS's `"凑四句一切"` option bundled 4–6 sentences (`17–21s`) into a single continuous pass, GPT drifted mid-way through without breath resets (the *"running out of oxygen"* effect).
   - Splitting at Japanese terminal punctuation (`。`, `！`, `？`) with `pause_second=0.45` keeps every chunk in the `3–8s` training distribution and inserts natural pauses between thoughts.
2. **Real-Time Factor (`RTF`) & Inference Optimization:**
   - `RTF = Generation Wall Time / Generated Audio Duration` (lower is faster; `< 1.0` means faster than real time).
   - On your RTX 5070 in FP16 with CUDA Graphs enabled, warmed-up `v2ProPlus` measured **`0.562–0.614` RTF** (~`1.6–1.8×` faster than real time).
   - First-chunk warmup compiles the CUDA graph (`~1–2s` once per model load), after which streaming chunk-by-chunk playback can keep up with real-time conversation.

---

## 6. How to Explain This Clearly in an Interview

If an interviewer asks *"What did you actually build here?"*, you can summarize it in 3 concrete layers:
1. **Data & Audio Engineering:** Scraped/filtered raw voice corpora, detected edge/noise/clipping failures with Silero VAD + spectral denoising, and matched strict multi-speaker cleanliness thresholds across `36+` hours of audio.
2. **Two-Stage Generative Modeling:** Fine-tuned both an autoregressive semantic Transformer (`GPT`, `s1`) and a VAE-GAN neural vocoder (`SoVITS`, `s2` with `ERes2NetV2` speaker conditioning) in mixed-precision PyTorch on Blackwell hardware (`sm_120` / CUDA 12.8).
3. **Inference Systems & Quality Tuning:** Diagnosed autoregressive length degradation and reference-prompt breath conditioning empirically via RMS/energy-frame analysis, achieving sub-real-time `32 kHz` synthesis (`~0.56–0.61` RTF) locally with `$0` API cost.

---

## 7. Is This Generative AI (GenAI)? And What About Modern VOCALOID?

### Is GPT-SoVITS Generative AI?
**Yes — 100% Generative AI.**
Unlike *discriminative* AI (which classifies inputs, e.g., "is this audio clip Ichika or Saki?"), **Generative AI** learns a probability distribution over real data and creates *brand-new* outputs that never existed in the recordings:
- **Stage 1 (`GPT`):** Autoregressive generative model (same Transformer family as text LLMs, generating new `25 Hz` speech token sequences).
- **Stage 2 (`SoVITS`):** Deep generative acoustic model (**VAE + Normalizing Flows + GAN Vocoder**), generating brand-new `32 kHz` audio waveforms from latent variables and semantic tokens.

*(By contrast, classic `VOCALOID2–4` was **not** GenAI — it was rule/algorithmic **concatenative synthesis**, slicing pre-recorded human diphone/triphone snippets from a database and stretching them with signal-processing algorithms like PSOLA.)*

### Is Recent VOCALOID (`VOCALOID6`) Also Generative AI?
**Yes.** Starting with **VOCALOID6** (`VOCALOID:AI`, Yamaha's neural singing engine — and similarly in modern singing synthesizers like Synthesizer V AI and DiffSinger):
- **Old VOCALOID (V2–V5 classic tracks):** Sample-stitching (`concatenative` / `PSOLA`) — gluing recorded phoneme slices together and reshaping pitch mathematically. Very distinctive "choppy/metallic" sound unless heavily hand-tuned.
- **VOCALOID6 AI voicebanks:** Deep neural networks trained on hours of a human vocalist's singing data. Given MIDI notes + lyrics, the neural network *generates* continuous pitch bends, vibrato, breathiness, and spectrograms/waveforms from scratch — making **VOCALOID6 AI** a specialized **Singing Voice Synthesis (SVS) GenAI** cousin of the **Speech TTS GenAI** (`GPT-SoVITS`) you're training here!
- **Key difference:** Your `GPT-SoVITS` model predicts rhythm/melody freely from text + punctuation (**speaking**), whereas `VOCALOID6` / SVS models are strictly constrained to exact musical **MIDI notes, tempo, and lyric timing** (**singing**).
