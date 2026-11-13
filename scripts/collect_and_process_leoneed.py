"""Index, download, and preprocess solo voice lines for Saki, Honami, and Shiho from cached Sekai assets."""
from __future__ import annotations

import concurrent.futures
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(r"e:\voiceSynth")
DB_DIR = ROOT / "data" / "sekai_cache" / "db"
ASSET_DIR = ROOT / "data" / "sekai_cache" / "assets"
BASE_ASSETS = "https://storage.sekai.best/sekai-jp-assets/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://sekai.best/",
}

CHARACTERS = {
    "saki": {
        "speaker": "咲希",
        "valid_2d_ids": {2, 279, 371, 459},
    },
    "honami": {
        "speaker": "穂波",
        "valid_2d_ids": {3, 280, 372, 391, 460},
    },
    "shiho": {
        "speaker": "志歩",
        "valid_2d_ids": {4, 281, 373, 461},
    },
}

JP_CHAR_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
FILLER_ONLY_RE = re.compile(r"^[ぁあいうえおっッんンえっあっうっふっはっ…・ー—―、。！？\s]+$")

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


def clean_sekai_text(text: str) -> str:
    t = text.replace("\r\n", "").replace("\n", "").replace("\u3000", "").strip()
    if t.startswith("（") and t.endswith("）") and t.count("（") == 1 and t.count("）") == 1:
        t = t[1:-1]
    t = re.sub(r"[—―]{2,}", "——", t)
    t = re.sub(r"…{2,}", "……", t)
    t = t.replace("『", "").replace("』", "").replace("「", "").replace("」", "")
    return t.strip()


def is_clean_text(text: str) -> bool:
    if not text:
        return False
    if text.count("…") >= 6 and len(JP_CHAR_RE.findall(text)) < 18:
        return False
    if len(JP_CHAR_RE.findall(text)) < 12:
        return False
    stripped = re.sub(r"[、。！？…・ー—―\s]", "", text)
    if FILLER_ONLY_RE.match(stripped):
        return False
    if any(ch in text for ch in ["<", ">", "{", "}", "[", "]", "#"]):
        return False
    return True


def build_voice_prefix_map() -> dict[tuple[str, str], str]:
    """Map (source_type, scenario_id) -> mp3 voice url prefix from cached DBs."""
    mapping: dict[tuple[str, str], str] = {}
    units = json.loads((DB_DIR / "unitStories.json").read_text(encoding="utf-8"))
    for u in units:
        for ch in u.get("chapters", []):
            for ep in ch.get("episodes", []):
                sid = ep["scenarioId"]
                mapping[("unit", sid)] = f"{BASE_ASSETS}sound/scenario/voice/{sid}/"
    card_eps = json.loads((DB_DIR / "cardEpisodes.json").read_text(encoding="utf-8"))
    for ep in card_eps:
        sid = ep["scenarioId"]
        mapping[("card", sid)] = f"{BASE_ASSETS}sound/card_scenario/voice/{sid}/"
    events = json.loads((DB_DIR / "eventStories.json").read_text(encoding="utf-8"))
    for ev in events:
        for ep in ev.get("eventStoryEpisodes", []):
            sid = ep["scenarioId"]
            mapping[("event", sid)] = f"{BASE_ASSETS}sound/scenario/voice/{sid}/"
    specials = json.loads((DB_DIR / "specialStories.json").read_text(encoding="utf-8"))
    for sp in specials:
        for ep in sp.get("episodes", []):
            sid = ep["scenarioId"]
            mapping[("special", sid)] = f"{BASE_ASSETS}sound/scenario/voice/{sid}/"
    return mapping


def index_characters() -> dict[str, list[dict]]:
    prefix_map = build_voice_prefix_map()
    by_char: dict[str, list[dict]] = {c: [] for c in CHARACTERS}
    seen_vids: dict[str, set[str]] = {c: set() for c in CHARACTERS}

    for src_type in ["unit", "card", "event", "special"]:
        src_dir = ASSET_DIR / src_type
        if not src_dir.exists():
            continue
        for asset_file in sorted(src_dir.glob("*.asset.json")):
            sid = asset_file.name[:-len(".asset.json")]
            prefix = prefix_map.get((src_type, sid))
            if not prefix:
                continue
            try:
                data = json.loads(asset_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            for td in data.get("TalkData", []):
                voices = td.get("Voices") or []
                if len(voices) != 1:
                    continue
                v = voices[0]
                c2d = v.get("Character2dId")
                speaker = (td.get("WindowDisplayName") or "").strip()
                vid = (v.get("VoiceId") or "").strip()
                if not vid or "partvoice" in vid.lower():
                    continue
                for char_key, cfg in CHARACTERS.items():
                    if c2d in cfg["valid_2d_ids"] and speaker == cfg["speaker"]:
                        if vid in seen_vids[char_key]:
                            break
                        cleaned = clean_sekai_text(td.get("Body") or "")
                        if not is_clean_text(cleaned):
                            break
                        seen_vids[char_key].add(vid)
                        by_char[char_key].append({
                            "voice_id": vid,
                            "source_type": src_type,
                            "scenario_id": sid,
                            "char2d_id": c2d,
                            "mp3_url": f"{prefix}{vid}.mp3",
                            "text": cleaned,
                        })
                        break

    for char_key, rows in by_char.items():
        char_dir = ROOT / "data" / char_key
        char_dir.mkdir(parents=True, exist_ok=True)
        cand_path = char_dir / "candidates.jsonl"
        with cand_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[index] {char_key}: {len(rows)} solo clean candidates -> {cand_path}")
    return by_char


def download_one(item: tuple[dict, Path]) -> str:
    row, raw_dir = item
    out = raw_dir / f"{row['voice_id']}.mp3"
    if out.exists() and out.stat().st_size > 1000:
        return "cached"
    for attempt in range(3):
        try:
            req = urllib.request.Request(row["mp3_url"], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            if len(data) > 500:
                out.write_bytes(data)
                return "ok"
            return "empty"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "404"
            time.sleep(0.4 * (attempt + 1))
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return "err"


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


def process_one(task: tuple[str, dict, str, str]) -> tuple[str, dict | None]:
    char_key, row, raw_dir_s, wav_dir_s = task
    vid = row["voice_id"]
    mp3_path = Path(raw_dir_s) / f"{vid}.mp3"
    wav_path = Path(wav_dir_s) / f"{vid}.wav"
    if not mp3_path.exists():
        return ("missing_mp3", None)

    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(mp3_path),
        "-af", "highpass=f=80:poles=2", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or len(res.stdout) < 4 * SR:
        return ("decode_err", None)
    wav = np.frombuffer(res.stdout, dtype=np.float32).copy()
    if int(np.sum(np.abs(wav) >= 0.995)) > 15:
        return ("clipped", None)

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

    speech_mask_32k = np.zeros(len(wav), dtype=bool)
    for seg in ts:
        speech_mask_32k[max(0, seg["start"] * 2) : min(len(wav), seg["end"] * 2)] = True
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

    speech_only = wav[speech_mask_32k]
    speech_rms = float(np.sqrt(np.mean(speech_only ** 2))) if len(speech_only) else 0.0
    if speech_rms < 0.012:
        return ("too_quiet", None)

    try:
        if len(noise_samples) >= int(0.12 * SR):
            trimmed = nr.reduce_noise(
                y=trimmed, sr=SR, y_noise=noise_samples[: SR * 2],
                stationary=True, prop_decrease=0.42, n_fft=1024,
                freq_mask_smooth_hz=500, time_mask_smooth_ms=50,
            ).astype(np.float32)
        else:
            trimmed = nr.reduce_noise(
                y=trimmed, sr=SR, stationary=True, prop_decrease=0.30,
                n_fft=1024, freq_mask_smooth_hz=500, time_mask_smooth_ms=50,
            ).astype(np.float32)
    except Exception:
        pass

    rel_segs = [(max(0, s["start"] * 2 - trim_start), min(len(trimmed), s["end"] * 2 - trim_start)) for s in ts]
    trimmed = soft_gate_gaps(trimmed, rel_segs)
    cosine_fade_edges(trimmed, int(0.035 * SR))

    peak = float(np.max(np.abs(trimmed)))
    if peak > 0.95:
        trimmed *= 0.95 / peak
        peak = 0.95
    elif 0.05 < peak < 0.30:
        trimmed *= 0.45 / peak
        peak = 0.45

    sf.write(str(wav_path), np.clip(trimmed, -1.0, 1.0), SR, subtype="PCM_16")
    return ("ok", {
        "voice_id": vid,
        "wav_path": str(wav_path),
        "text": row["text"],
        "source_type": row["source_type"],
        "scenario_id": row["scenario_id"],
        "duration": round(dur, 4),
        "voiced_ratio": round(voiced_ratio, 4),
        "speech_rms": round(speech_rms, 5),
        "peak": round(peak, 4),
    })


def main() -> None:
    t0 = time.time()
    by_char = index_characters()

    # 1. Download MP3s for all 3 characters
    for char_key, rows in by_char.items():
        raw_dir = ROOT / "data" / char_key / "raw_mp3"
        raw_dir.mkdir(parents=True, exist_ok=True)
        stats: dict[str, int] = {}
        t_dl = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            for i, st in enumerate(pool.map(download_one, [(r, raw_dir) for r in rows]), 1):
                stats[st] = stats.get(st, 0) + 1
                if i % 1500 == 0 or i == len(rows):
                    print(f"[dl:{char_key} {i}/{len(rows)}] {stats} ({time.time() - t_dl:.1f}s)")

    # 2. Preprocess into clean 32kHz mono WAVs (no 4h/8h split yet)
    for char_key, rows in by_char.items():
        raw_dir = ROOT / "data" / char_key / "raw_mp3"
        wav_dir = ROOT / "data" / char_key / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        tasks = [(char_key, r, str(raw_dir), str(wav_dir)) for r in rows]
        stats: dict[str, int] = {}
        records: list[dict] = []
        total_dur = 0.0
        t_pp = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=6, initializer=_init_worker) as pool:
            for i, (st, rec) in enumerate(pool.map(process_one, tasks, chunksize=16), 1):
                stats[st] = stats.get(st, 0) + 1
                if rec is not None:
                    records.append(rec)
                    total_dur += rec["duration"]
                if i % 1000 == 0 or i == len(rows):
                    print(f"[pp:{char_key} {i}/{len(rows)}] ok={len(records)} ({total_dur / 3600.0:.2f}h) stats={stats} ({time.time() - t_pp:.1f}s)")

        proc_jsonl = ROOT / "data" / char_key / "processed.jsonl"
        with proc_jsonl.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_list = ROOT / "data" / char_key / "all.list"
        all_list.write_text(
            "\n".join(f"{r['wav_path']}|{char_key}|ja|{r['text']}" for r in records) + "\n",
            encoding="utf-8",
        )
        print(f"[done:{char_key}] {len(records)} clean WAVs ({total_dur / 3600.0:.2f}h) -> {all_list}")

    print(f"All 3 characters finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
