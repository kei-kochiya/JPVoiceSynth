"""Scan Project SEKAI master DB & scenario assets to index all solo Hoshino Ichika voice candidates."""
from __future__ import annotations

import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DB = "https://sekai-world.github.io/sekai-master-db-diff/"
BASE_ASSETS = "https://storage.sekai.best/sekai-jp-assets/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://sekai.best/",
}

# Adult/teen Ichika Character2dIds from character2ds.json (excludes childhood 352/488, mob 32, collab 900/906)
VALID_ICHIKA_2D_IDS = {1, 278, 370, 458}

ROOT = Path(r"e:\voiceSynth")
CACHE_DIR = ROOT / "data" / "sekai_cache"
DB_DIR = CACHE_DIR / "db"
ASSET_DIR = CACHE_DIR / "assets"
OUT_MANIFEST = ROOT / "data" / "ichika_candidates.jsonl"


def fetch_bytes(url: str, retries: int = 3, timeout: int = 15) -> bytes | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
    return None


def fetch_json_cached(url: str, cache_path: Path) -> dict | list | None:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    raw = fetch_bytes(url)
    if raw is None:
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(raw)
    return json.loads(raw.decode("utf-8"))


JP_CHAR_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
FILLER_ONLY_RE = re.compile(r"^[ぁあいうえおっッんンえっあっうっふっはっ…・ー—―、。！？\s]+$")


def clean_sekai_text(text: str) -> str:
    # Replace newlines and fullwidth spaces
    t = text.replace("\r\n", "").replace("\n", "").replace("\u3000", "")
    # Remove outer thought parentheses if wrapping the whole line
    t = t.strip()
    if t.startswith("（") and t.endswith("）") and t.count("（") == 1 and t.count("）") == 1:
        t = t[1:-1]
    # Normalize multiple dashes/ellipses slightly without losing prosody
    t = re.sub(r"[—―]{2,}", "——", t)
    t = re.sub(r"…{2,}", "……", t)
    # Remove stray quotation brackets that break G2P pauses if unmatched
    t = t.replace("『", "").replace("』", "").replace("「", "").replace("」", "")
    return t.strip()


def is_clean_text(text: str) -> bool:
    if not text:
        return False
    # Reject lines with excessive ellipses or pure hesitation
    if text.count("…") >= 6 and len(JP_CHAR_RE.findall(text)) < 18:
        return False
    jp_chars = JP_CHAR_RE.findall(text)
    if len(jp_chars) < 12:
        return False
    stripped = re.sub(r"[、。！？…・ー—―\s]", "", text)
    if FILLER_ONLY_RE.match(stripped):
        return False
    # Reject lines with weird symbols/ASCII placeholders
    if any(ch in text for ch in ["<", ">", "{", "}", "[", "]", "#"]):
        return False
    return True


def build_episode_tasks() -> list[tuple[str, str, str, str]]:
    """Return list of (source_type, scenario_id, asset_url, voice_url_prefix)."""
    tasks: list[tuple[str, str, str, str]] = []
    seen_scenarios: set[tuple[str, str]] = set()

    # 1. Unit stories (scan all chapters, prioritizing light_sound first)
    units = fetch_json_cached(BASE_DB + "unitStories.json", DB_DIR / "unitStories.json") or []
    units_sorted = sorted(units, key=lambda u: 0 if u.get("unit") == "light_sound" else 1)
    for u in units_sorted:
        for ch in u.get("chapters", []):
            ch_ab = ch["assetbundleName"]
            for ep in ch.get("episodes", []):
                sid = ep["scenarioId"]
                key = ("unit", sid)
                if key in seen_scenarios:
                    continue
                seen_scenarios.add(key)
                tasks.append((
                    "unit",
                    sid,
                    f"{BASE_ASSETS}scenario/unitstory/{ch_ab}/{sid}.asset",
                    f"{BASE_ASSETS}sound/scenario/voice/{sid}/",
                ))

    # 2. Card episodes (prioritize Ichika cards first, then Leo/need, then others)
    cards = fetch_json_cached(BASE_DB + "cards.json", DB_DIR / "cards.json") or []
    card_char = {c["id"]: c.get("characterId") for c in cards}
    card_eps = fetch_json_cached(BASE_DB + "cardEpisodes.json", DB_DIR / "cardEpisodes.json") or []
    # Sort: Ichika cards (char 1) first, then Saki/Honami/Shiho (2,3,4), then others
    def card_prio(ep: dict) -> int:
        cid = card_char.get(ep.get("cardId"), 99)
        if cid == 1:
            return 0
        if cid in {2, 3, 4}:
            return 1
        return 2

    for ep in sorted(card_eps, key=card_prio):
        sid = ep["scenarioId"]
        ab = ep["assetbundleName"]
        key = ("card", sid)
        if key in seen_scenarios:
            continue
        seen_scenarios.add(key)
        tasks.append((
            "card",
            sid,
            f"{BASE_ASSETS}character/member/{ab}/{sid}.asset",
            f"{BASE_ASSETS}sound/card_scenario/voice/{sid}/",
        ))

    # 3. Event stories (prioritize Leo/need events first by bannerGameCharacterUnitId <= 4)
    events = fetch_json_cached(BASE_DB + "eventStories.json", DB_DIR / "eventStories.json") or []
    events_sorted = sorted(events, key=lambda e: 0 if e.get("bannerGameCharacterUnitId", 99) <= 4 else 1)
    for ev in events_sorted:
        ev_ab = ev["assetbundleName"]
        for ep in ev.get("eventStoryEpisodes", []):
            sid = ep["scenarioId"]
            key = ("event", sid)
            if key in seen_scenarios:
                continue
            seen_scenarios.add(key)
            tasks.append((
                "event",
                sid,
                f"{BASE_ASSETS}event_story/{ev_ab}/scenario/{sid}.asset",
                f"{BASE_ASSETS}sound/scenario/voice/{sid}/",
            ))

    # 4. Special stories
    specials = fetch_json_cached(BASE_DB + "specialStories.json", DB_DIR / "specialStories.json") or []
    for sp in specials:
        sp_ab = sp["assetbundleName"]
        for ep in sp.get("episodes", []):
            sid = ep["scenarioId"]
            key = ("special", sid)
            if key in seen_scenarios:
                continue
            seen_scenarios.add(key)
            tasks.append((
                "special",
                sid,
                f"{BASE_ASSETS}scenario/special/{sp_ab}/{sid}.asset",
                f"{BASE_ASSETS}sound/scenario/voice/{sid}/",
            ))

    return tasks


def process_episode(task: tuple[str, str, str, str]) -> list[dict]:
    src_type, sid, asset_url, voice_prefix = task
    cache_path = ASSET_DIR / src_type / f"{sid}.asset.json"
    data = fetch_json_cached(asset_url, cache_path)
    if not isinstance(data, dict):
        return []
    out: list[dict] = []
    for td in data.get("TalkData", []):
        voices = td.get("Voices") or []
        if len(voices) != 1:
            continue
        v = voices[0]
        if v.get("Character2dId") not in VALID_ICHIKA_2D_IDS:
            continue
        speaker = (td.get("WindowDisplayName") or "").strip()
        if speaker != "一歌":
            continue
        vid = (v.get("VoiceId") or "").strip()
        if not vid or "partvoice" in vid.lower():
            continue
        raw_body = td.get("Body") or ""
        cleaned = clean_sekai_text(raw_body)
        if not is_clean_text(cleaned):
            continue
        out.append({
            "voice_id": vid,
            "source_type": src_type,
            "scenario_id": sid,
            "char2d_id": v.get("Character2dId"),
            "mp3_url": f"{voice_prefix}{vid}.mp3",
            "text": cleaned,
            "raw_text": raw_body,
        })
    return out


def main() -> None:
    t0 = time.time()
    tasks = build_episode_tasks()
    print(f"Total unique scenario episodes to scan: {len(tasks)}")

    all_rows: list[dict] = []
    seen_voice_ids: set[str] = set()
    counts_by_type: dict[str, int] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for idx, rows in enumerate(pool.map(process_episode, tasks), 1):
            for r in rows:
                vid = r["voice_id"]
                if vid in seen_voice_ids:
                    continue
                seen_voice_ids.add(vid)
                all_rows.append(r)
                counts_by_type[r["source_type"]] = counts_by_type.get(r["source_type"], 0) + 1
            if idx % 500 == 0 or idx == len(tasks):
                print(f"[{idx}/{len(tasks)}] found {len(all_rows)} clean solo Ichika candidates ({counts_by_type}) in {time.time() - t0:.1f}s")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} candidates to {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
