"""Concurrently download Hoshino Ichika candidate MP3s from storage.sekai.best."""
from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

MANIFEST = Path(r"e:\voiceSynth\data\ichika_candidates.jsonl")
RAW_DIR = Path(r"e:\voiceSynth\data\raw_mp3")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://sekai.best/",
}


def download_one(row: dict) -> tuple[str, str]:
    vid = row["voice_id"]
    out = RAW_DIR / f"{vid}.mp3"
    if out.exists() and out.stat().st_size > 1000:
        return ("cached", vid)
    url = row["mp3_url"]
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            if len(data) > 500:
                out.write_bytes(data)
                return ("ok", vid)
            return ("empty", vid)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ("404", vid)
            time.sleep(0.4 * (attempt + 1))
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return ("err", vid)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Downloading up to {len(rows)} Ichika MP3s into {RAW_DIR}...")
    t0 = time.time()
    stats: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for i, (status, _) in enumerate(pool.map(download_one, rows), 1):
            stats[status] = stats.get(status, 0) + 1
            if i % 1000 == 0 or i == len(rows):
                print(f"[{i}/{len(rows)}] {stats} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
