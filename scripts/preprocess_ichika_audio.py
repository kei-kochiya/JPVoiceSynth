"""Preprocess Ichika MP3s into clean 32kHz mono WAVs with anti-breath & anti-noise filtering."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(r"e:\voiceSynth")
MANIFEST = ROOT / "data" / "ichika_candidates.jsonl"
RAW_DIR = ROOT / "data" / "raw_mp3"
WAV_DIR = ROOT / "data" / "ichika_wavs"
PROCESSED_JSONL = ROOT / "data" / "ichika_processed.jsonl"
LIST_4H = ROOT / "data" / "ichika_4h.list"
LIST_8H = ROOT / "data" / "ichika_8h.list"

SR = 32000
SR_VAD = 16000
MIN_DUR = 2.0
MAX_DUR = 12.0
MIN_VOICED_RATIO = 0.55

_VAD_MODEL = None


def _init_worker() -> None:
    global _VAD_MODEL
    torch.set_num_threads(1)
    _VAD_MODEL = load_silero_vad()


def decode_mp3_hpf(mp3_path: Path) -> np.ndarray | None:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        "-af",
        "highpass=f=80:poles=2",
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or len(res.stdout) < 4 * SR:  # < 1s raw audio
        return None
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def cosine_fade_edges(wav: np.ndarray, fade_samples: int) -> None:
    n = min(fade_samples, len(wav) // 4)
    if n <= 1:
        return
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]


def soft_gate_gaps(
    wav: np.ndarray,
    speech_segs_32k: list[tuple[int, int]],
    gap_gain: float = 0.125,
    min_gap_samples: int = int(0.16 * SR),
    ramp_samples: int = int(0.025 * SR),
) -> np.ndarray:
    """Attenuate inter-word pauses/breaths gently instead of hard-muting to zero."""
    if len(speech_segs_32k) <= 1:
        return wav
    gain_env = np.ones(len(wav), dtype=np.float32)
    for idx in range(len(speech_segs_32k) - 1):
        g_start = speech_segs_32k[idx][1]
        g_end = speech_segs_32k[idx + 1][0]
        if g_end - g_start < min_gap_samples:
            continue
        r = min(ramp_samples, (g_end - g_start) // 3)
        if r > 1:
            down = 1.0 - (1.0 - gap_gain) * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, r, dtype=np.float32)))
            up = gap_gain + (1.0 - gap_gain) * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, r, dtype=np.float32)))
            gain_env[g_start : g_start + r] = down
            gain_env[g_start + r : g_end - r] = gap_gain
            gain_env[g_end - r : g_end] = up
        else:
            gain_env[g_start:g_end] = gap_gain
    return wav * gain_env


def process_one(row: dict) -> tuple[str, dict | None]:
    vid = row["voice_id"]
    mp3_path = RAW_DIR / f"{vid}.mp3"
    wav_path = WAV_DIR / f"{vid}.wav"

    if not mp3_path.exists():
        return ("missing_mp3", None)

    wav = decode_mp3_hpf(mp3_path)
    if wav is None or len(wav) == 0:
        return ("decode_err", None)

    # Check raw clipping
    if int(np.sum(np.abs(wav) >= 0.995)) > 15:
        return ("clipped", None)

    # 16kHz copy for Silero VAD
    wav16 = resample_poly(wav, 1, 2).astype(np.float32)
    ts = get_speech_timestamps(
        torch.from_numpy(wav16),
        _VAD_MODEL,
        sampling_rate=SR_VAD,
        threshold=0.45,
        min_speech_duration_ms=250,
        min_silence_duration_ms=220,
        speech_pad_ms=65,
    )
    if not ts:
        return ("no_speech", None)

    # Collect non-speech noise profile from original clip (before trimming) if >= 0.12s exists
    speech_mask_32k = np.zeros(len(wav), dtype=bool)
    for seg in ts:
        s32 = max(0, seg["start"] * 2)
        e32 = min(len(wav), seg["end"] * 2)
        speech_mask_32k[s32:e32] = True

    noise_samples = wav[~speech_mask_32k]

    trim_start = max(0, ts[0]["start"] * 2)
    trim_end = min(len(wav), ts[-1]["end"] * 2)
    trimmed = wav[trim_start:trim_end].copy()

    dur = len(trimmed) / SR
    if dur < MIN_DUR:
        return ("too_short", None)
    if dur > MAX_DUR:
        return ("too_long", None)

    voiced_samples = sum((seg["end"] - seg["start"]) * 2 for seg in ts)
    voiced_ratio = min(1.0, voiced_samples / max(1, len(trimmed)))
    if voiced_ratio < MIN_VOICED_RATIO:
        return ("low_voiced_ratio", None)

    # Measure speech RMS on voiced regions
    speech_only = wav[speech_mask_32k]
    speech_rms = float(np.sqrt(np.mean(speech_only ** 2))) if len(speech_only) else 0.0
    if speech_rms < 0.012:
        return ("too_quiet", None)

    # Gentle stationary spectral denoise using measured floor (or stationary estimate)
    try:
        if len(noise_samples) >= int(0.12 * SR):
            trimmed = nr.reduce_noise(
                y=trimmed,
                sr=SR,
                y_noise=noise_samples[: SR * 2],
                stationary=True,
                prop_decrease=0.42,
                n_fft=1024,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
            ).astype(np.float32)
        else:
            trimmed = nr.reduce_noise(
                y=trimmed,
                sr=SR,
                stationary=True,
                prop_decrease=0.30,
                n_fft=1024,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
            ).astype(np.float32)
    except Exception:
        pass

    # Soft-gate mid-clip breaths/pauses (>160ms non-speech gaps)
    rel_segs = [(max(0, s["start"] * 2 - trim_start), min(len(trimmed), s["end"] * 2 - trim_start)) for s in ts]
    trimmed = soft_gate_gaps(trimmed, rel_segs)

    # Cosine fade-in / fade-out (35ms)
    cosine_fade_edges(trimmed, int(0.035 * SR))

    # Gentle peak guard (avoid boosting quiet noise floors)
    peak = float(np.max(np.abs(trimmed)))
    if peak > 0.95:
        trimmed *= 0.95 / peak
        peak = 0.95
    elif 0.05 < peak < 0.30:
        trimmed *= 0.45 / peak
        peak = 0.45

    sf.write(str(wav_path), np.clip(trimmed, -1.0, 1.0), SR, subtype="PCM_16")

    return (
        "ok",
        {
            "voice_id": vid,
            "wav_path": str(wav_path),
            "text": row["text"],
            "source_type": row["source_type"],
            "scenario_id": row["scenario_id"],
            "duration": round(dur, 4),
            "voiced_ratio": round(voiced_ratio, 4),
            "speech_rms": round(speech_rms, 5),
            "peak": round(peak, 4),
        },
    )


def write_lists(records: list[dict]) -> None:
    # Keep natural order (unit -> ichika card -> leo/need events -> other events)
    # Write 4h and 8h .list files
    for target_hours, out_path in [(4.0, LIST_4H), (8.0, LIST_8H)]:
        target_sec = target_hours * 3600.0
        acc = 0.0
        lines = []
        for r in records:
            if acc >= target_sec:
                break
            lines.append(f"{r['wav_path']}|ichika|ja|{r['text']}")
            acc += r["duration"]
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} lines ({acc / 3600.0:.2f} hours) -> {out_path}")


def main() -> None:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Preprocessing {len(rows)} Ichika candidates (2.0s..12.0s, HPF 80Hz, Silero VAD, gentle denoise + breath gate)...")

    t0 = time.time()
    stats: dict[str, int] = {}
    records: list[dict] = []
    total_dur = 0.0

    with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=_init_worker) as pool:
        for i, (status, rec) in enumerate(pool.map(process_one, rows, chunksize=16), 1):
            stats[status] = stats.get(status, 0) + 1
            if rec is not None:
                records.append(rec)
                total_dur += rec["duration"]
            if i % 500 == 0 or i == len(rows):
                print(
                    f"[{i}/{len(rows)}] ok={len(records)} ({total_dur / 3600.0:.2f}h) "
                    f"stats={stats} ({time.time() - t0:.1f}s)"
                )

    with PROCESSED_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_lists(records)


if __name__ == "__main__":
    main()
