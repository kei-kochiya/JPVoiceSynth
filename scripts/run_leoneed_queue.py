"""Sequential full-dataset GPT-SoVITS v2ProPlus trainer for Ichika, Saki, Honami, and Shiho."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from run_gptsovits_pipeline import PYTHON_EXE, REPO_DIR, ensure_users_pth, run_prepare, run_s1, run_s2

JOBS = [
    ("ichika", "ichika_full", Path(r"e:\voiceSynth\data\ichika_full.list")),
    ("saki", "saki_full", Path(r"e:\voiceSynth\data\saki\all.list")),
    ("honami", "honami_full", Path(r"e:\voiceSynth\data\honami\all.list")),
    ("shiho", "shiho_full", Path(r"e:\voiceSynth\data\shiho\all.list")),
]

SOVITS_DIR = REPO_DIR / "SoVITS_weights_v2ProPlus"
GPT_DIR = REPO_DIR / "GPT_weights_v2ProPlus"
REFS_JSON = Path(r"e:\voiceSynth\references\references.json")
SAMPLE_OUT_DIR = Path(r"e:\voiceSynth\outputs\leoneed_samples")
SYNTH_SCRIPT = Path(r"e:\voiceSynth\scripts\synthesize.py")


def query_gpu_0() -> tuple[int, int, list[str]]:
    """Return (free_mib, util_pct, heavy_compute_apps) for GPU 0."""
    res = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not res.stdout.strip():
        return (0, 100, ["nvidia-smi_unavailable"])
    parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
    free_mib = int(float(parts[0]))
    util_pct = int(float(parts[1]))

    apps_res = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    heavy_apps: list[str] = []
    if apps_res.returncode == 0:
        for line in apps_res.stdout.strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 3:
                try:
                    mem = float(cols[2])
                    if mem > 300:
                        heavy_apps.append(f"{cols[1]}(pid={cols[0]},{int(mem)}MiB)")
                except ValueError:
                    pass
    return free_mib, util_pct, heavy_apps


def wait_for_gpu_free(
    stage_label: str,
    min_free_mib: int = 8500,
    max_util_pct: int = 25,
    poll_sec: int = 30,
    stable_checks: int = 3,
) -> None:
    """Block until GPU 0 has ample free VRAM and low utilization across consecutive checks."""
    streak = 0
    while streak < stable_checks:
        free_mib, util_pct, heavy_apps = query_gpu_0()
        is_free = (free_mib >= min_free_mib) and (util_pct <= max_util_pct) and (not heavy_apps)
        if is_free:
            streak += 1
            if streak < stable_checks:
                time.sleep(2)
        else:
            streak = 0
            print(
                f"[gpu-wait:{stage_label}] Busy: free={free_mib}MiB (need>={min_free_mib}), "
                f"util={util_pct}% (need<={max_util_pct}%), heavy={heavy_apps}. Sleeping {poll_sec}s..."
            )
            time.sleep(poll_sec)
    print(f"[gpu-ready:{stage_label}] GPU 0 ready (free={free_mib}MiB, util={util_pct}%).")


def cleanup_intermediate_logs(exp_name: str) -> None:
    """Remove bulky feature caches and intermediate D_/G_ checkpoints after weights are exported."""
    exp_dir = REPO_DIR / "logs" / exp_name
    for sub in ["4-cnhubert", "5-wav32k", "7-sv_cn"]:
        tp = exp_dir / sub
        if tp.exists():
            shutil.rmtree(tp, ignore_errors=True)
    s2_logs = exp_dir / "logs_s2_v2ProPlus"
    if s2_logs.exists():
        for ckpt in list(s2_logs.glob("G_*.pth")) + list(s2_logs.glob("D_*.pth")):
            ckpt.unlink(missing_ok=True)


def run_sample_check(char_key: str, exp_name: str, s2_epochs: int, s1_epochs: int) -> None:
    """Synthesize a quick multi-sentence verification sample for a finished character."""
    if not REFS_JSON.exists():
        return
    refs = json.loads(REFS_JSON.read_text(encoding="utf-8"))
    if char_key not in refs:
        return
    sovits_matches = sorted(SOVITS_DIR.glob(f"{exp_name}_e{s2_epochs}_*.pth"))
    gpt_ckpt = GPT_DIR / f"{exp_name}-e{s1_epochs}.ckpt"
    if not sovits_matches or not gpt_ckpt.exists():
        return
    SAMPLE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_wav = SAMPLE_OUT_DIR / f"{exp_name}_sample.wav"
    text = (
        "今日も放課後にみんなでスタジオへ集まって、新曲の練習をしようと思ってるんだ。"
        "最初は少し難しいフレーズもあったけど、何回も合わせていくうちに息が合ってきた気がするよ！"
    )
    subprocess.run(
        [
            str(PYTHON_EXE),
            "-s",
            str(SYNTH_SCRIPT),
            "--gpt_model",
            str(gpt_ckpt),
            "--sovits_model",
            str(sovits_matches[-1]),
            "--ref_audio",
            refs[char_key]["ref_audio"],
            "--ref_text",
            refs[char_key]["ref_text"],
            "--text",
            text,
            "--out",
            str(out_wav),
        ],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s2_batch", type=int, default=6)
    parser.add_argument("--s2_epochs", type=int, default=6)
    parser.add_argument("--s2_save_every", type=int, default=3)
    parser.add_argument("--s1_batch", type=int, default=6)
    parser.add_argument("--s1_epochs", type=int, default=10)
    parser.add_argument("--s1_save_every", type=int, default=5)
    parser.add_argument("--n_parts", type=int, default=2)
    args = parser.parse_args()

    ensure_users_pth()
    queue_summary: dict[str, dict] = {}
    summary_path = Path(r"e:\voiceSynth\data\leoneed_queue_summary.json")

    for char_key, exp_name, list_path in JOBS:
        if not list_path.exists():
            raise FileNotFoundError(f"Missing dataset list: {list_path}")
        line_count = len([l for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()])
        print(f"\n============================================================")
        print(f"[QUEUE] Starting {char_key} ({exp_name}): {line_count} lines")
        print(f"============================================================")

        final_s2 = sorted(SOVITS_DIR.glob(f"{exp_name}_e{args.s2_epochs}_*.pth"))
        final_s1 = GPT_DIR / f"{exp_name}-e{args.s1_epochs}.ckpt"
        if final_s2 and final_s1.exists():
            print(f"[QUEUE] {exp_name} already has final weights ({final_s2[-1].name}, {final_s1.name}); skipping.")
            continue

        t_char = time.time()
        wait_for_gpu_free(f"{exp_name}:prepare")
        timings = run_prepare(exp_name, list_path, n_parts=args.n_parts)

        if not final_s2:
            wait_for_gpu_free(f"{exp_name}:s2_train")
            timings["s2_train"] = run_s2(exp_name, args.s2_batch, args.s2_epochs, args.s2_save_every)
        else:
            timings["s2_train"] = 0.0

        if not final_s1.exists():
            wait_for_gpu_free(f"{exp_name}:s1_train")
            timings["s1_train"] = run_s1(exp_name, args.s1_batch, args.s1_epochs, args.s1_save_every)
        else:
            timings["s1_train"] = 0.0

        timings["total_sec"] = time.time() - t_char
        timings["total_min"] = timings["total_sec"] / 60.0
        timings["lines"] = line_count

        Path(rf"e:\voiceSynth\data\benchmark_{exp_name}.json").write_text(
            json.dumps(timings, indent=2), encoding="utf-8"
        )
        queue_summary[exp_name] = timings
        summary_path.write_text(json.dumps(queue_summary, indent=2), encoding="utf-8")

        cleanup_intermediate_logs(exp_name)
        run_sample_check(char_key, exp_name, args.s2_epochs, args.s1_epochs)
        print(f"[QUEUE] Finished {exp_name} in {timings['total_min']:.2f} min.")

    print("\n[QUEUE] ALL 4 CHARACTERS COMPLETE.")


if __name__ == "__main__":
    main()
