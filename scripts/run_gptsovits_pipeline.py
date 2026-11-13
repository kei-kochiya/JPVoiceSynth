"""End-to-end GPT-SoVITS v2ProPlus preprocessing + training runner with per-stage timing."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_DIR = Path(r"e:\voiceSynth\GPT-SoVITS")
PYTHON_EXE = REPO_DIR / ".venv" / "Scripts" / "python.exe"


def run_cmd(cmd: list[str], env: dict[str, str], label: str) -> float:
    print(f"\n=== [{label}] START ===")
    print("CMD:", " ".join(cmd))
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO_DIR), env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"[{label}] failed with exit code {res.returncode} after {elapsed:.1f}s")
    print(f"=== [{label}] DONE in {elapsed:.1f}s ({elapsed / 60.0:.2f} min) ===")
    return elapsed


def ensure_users_pth() -> None:
    sp = REPO_DIR / ".venv" / "Lib" / "site-packages"
    pth = sp / "users.pth"
    paths = [
        str(REPO_DIR),
        str(REPO_DIR / "GPT_SoVITS" / "BigVGAN"),
        str(REPO_DIR / "tools"),
        str(REPO_DIR / "tools" / "asr"),
        str(REPO_DIR / "GPT_SoVITS"),
        str(REPO_DIR / "tools" / "uvr5"),
    ]
    pth.write_text("\n".join(paths) + "\n", encoding="utf-8")


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["version"] = "v2ProPlus"
    env["is_half"] = "True"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["no_proxy"] = "localhost, 127.0.0.1, ::1"
    env["PYTHONPATH"] = os.pathsep.join([
        str(REPO_DIR),
        str(REPO_DIR / "GPT_SoVITS"),
        str(REPO_DIR / "GPT_SoVITS" / "BigVGAN"),
        str(REPO_DIR / "tools"),
    ])
    return env


def run_prepare(exp_name: str, list_path: Path, n_parts: int = 2) -> dict[str, float]:
    opt_dir = REPO_DIR / "logs" / exp_name
    opt_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # 1A: Text + BERT
    path_text = opt_dir / "2-name2text.txt"
    if not path_text.exists() or len(path_text.read_text(encoding="utf-8").strip().splitlines()) < 2:
        t0 = time.time()
        procs = []
        for i_part in range(n_parts):
            env = base_env()
            env.update({
                "inp_text": str(list_path),
                "inp_wav_dir": "",
                "exp_name": exp_name,
                "opt_dir": str(opt_dir),
                "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                "i_part": str(i_part),
                "all_parts": str(n_parts),
                "_CUDA_VISIBLE_DEVICES": "0",
            })
            procs.append(subprocess.Popen(
                [str(PYTHON_EXE), "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"],
                cwd=str(REPO_DIR),
                env=env,
            ))
        for p in procs:
            if p.wait() != 0:
                raise RuntimeError("1-get-text.py failed")
        merged = []
        for i_part in range(n_parts):
            part_f = opt_dir / f"2-name2text-{i_part}.txt"
            content = part_f.read_text(encoding="utf-8").strip("\n")
            if content:
                merged.extend(content.split("\n"))
            part_f.unlink()
        path_text.write_text("\n".join(merged) + "\n", encoding="utf-8")
        timings["1a_text"] = time.time() - t0
        print(f"[1a_text] {len(merged)} items in {timings['1a_text']:.1f}s")
    else:
        timings["1a_text"] = 0.0

    # 1B: CNHubert + wav32k
    t0 = time.time()
    procs = []
    for i_part in range(n_parts):
        env = base_env()
        env.update({
            "inp_text": str(list_path),
            "inp_wav_dir": "",
            "exp_name": exp_name,
            "opt_dir": str(opt_dir),
            "cnhubert_base_dir": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "sv_path": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
            "i_part": str(i_part),
            "all_parts": str(n_parts),
            "_CUDA_VISIBLE_DEVICES": "0",
        })
        procs.append(subprocess.Popen(
            [str(PYTHON_EXE), "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"],
            cwd=str(REPO_DIR),
            env=env,
        ))
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError("2-get-hubert-wav32k.py failed")
    timings["1b_hubert_wav32k"] = time.time() - t0
    print(f"[1b_hubert_wav32k] done in {timings['1b_hubert_wav32k']:.1f}s")

    # 1B-sv: ERes2NetV2 speaker embeddings for v2ProPlus
    t0 = time.time()
    procs = []
    for i_part in range(n_parts):
        env = base_env()
        env.update({
            "inp_text": str(list_path),
            "inp_wav_dir": "",
            "exp_name": exp_name,
            "opt_dir": str(opt_dir),
            "sv_path": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
            "i_part": str(i_part),
            "all_parts": str(n_parts),
            "_CUDA_VISIBLE_DEVICES": "0",
        })
        procs.append(subprocess.Popen(
            [str(PYTHON_EXE), "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"],
            cwd=str(REPO_DIR),
            env=env,
        ))
    for p in procs:
        if p.wait() != 0:
            raise RuntimeError("2-get-sv.py failed")
    timings["1b_sv"] = time.time() - t0
    print(f"[1b_sv] done in {timings['1b_sv']:.1f}s")

    # 1C: Semantic tokens
    path_sem = opt_dir / "6-name2semantic.tsv"
    if not path_sem.exists() or path_sem.stat().st_size < 31:
        t0 = time.time()
        procs = []
        for i_part in range(n_parts):
            env = base_env()
            env.update({
                "inp_text": str(list_path),
                "exp_name": exp_name,
                "opt_dir": str(opt_dir),
                "pretrained_s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
                "s2config_path": "GPT_SoVITS/configs/s2v2ProPlus.json",
                "i_part": str(i_part),
                "all_parts": str(n_parts),
                "_CUDA_VISIBLE_DEVICES": "0",
            })
            procs.append(subprocess.Popen(
                [str(PYTHON_EXE), "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
                cwd=str(REPO_DIR),
                env=env,
            ))
        for p in procs:
            if p.wait() != 0:
                raise RuntimeError("3-get-semantic.py failed")
        opt = ["item_name\tsemantic_audio"]
        for i_part in range(n_parts):
            sem_part = opt_dir / f"6-name2semantic-{i_part}.tsv"
            content = sem_part.read_text(encoding="utf-8").strip("\n")
            if content:
                opt.extend(content.split("\n"))
            sem_part.unlink()
        path_sem.write_text("\n".join(opt) + "\n", encoding="utf-8")
        timings["1c_semantic"] = time.time() - t0
        print(f"[1c_semantic] {len(opt) - 1} items in {timings['1c_semantic']:.1f}s")
    else:
        timings["1c_semantic"] = 0.0

    return timings


def run_s2(exp_name: str, batch_size: int, epochs: int, save_every_epoch: int) -> float:
    s2_dir = REPO_DIR / "logs" / exp_name
    (s2_dir / "logs_s2_v2ProPlus").mkdir(parents=True, exist_ok=True)
    save_dir = REPO_DIR / "SoVITS_weights_v2ProPlus"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((REPO_DIR / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json").read_text(encoding="utf-8"))
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["epochs"] = epochs
    cfg["train"]["fp16_run"] = True
    cfg["train"]["text_low_lr_rate"] = 0.4
    cfg["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
    cfg["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth"
    cfg["train"]["if_save_latest"] = True
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["save_every_epoch"] = save_every_epoch
    cfg["train"]["gpu_numbers"] = "0"
    cfg["train"]["grad_ckpt"] = False
    cfg["train"]["lora_rank"] = 32
    cfg["model"]["version"] = "v2ProPlus"
    cfg["data"]["exp_dir"] = cfg["s2_ckpt_dir"] = str(s2_dir)
    cfg["save_weight_dir"] = "SoVITS_weights_v2ProPlus"
    cfg["name"] = exp_name
    cfg["version"] = "v2ProPlus"

    tmp_cfg = s2_dir / "s2_run_config.json"
    tmp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env = base_env()
    return run_cmd(
        [str(PYTHON_EXE), "-s", "GPT_SoVITS/s2_train.py", "--config", str(tmp_cfg)],
        env=env,
        label=f"s2_train ({exp_name}, bs={batch_size}, ep={epochs})",
    )


def run_s1(exp_name: str, batch_size: int, epochs: int, save_every_epoch: int) -> float:
    s1_dir = REPO_DIR / "logs" / exp_name
    (s1_dir / "logs_s1_v2ProPlus").mkdir(parents=True, exist_ok=True)
    save_dir = REPO_DIR / "GPT_weights_v2ProPlus"
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((REPO_DIR / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml").read_text(encoding="utf-8"))
    cfg["train"]["batch_size"] = batch_size
    cfg["train"]["epochs"] = epochs
    cfg["train"]["precision"] = "16-mixed"
    cfg["train"]["save_every_n_epoch"] = save_every_epoch
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["if_save_latest"] = True
    cfg["train"]["if_dpo"] = False
    cfg["train"]["half_weights_save_dir"] = "GPT_weights_v2ProPlus"
    cfg["train"]["exp_name"] = exp_name
    cfg["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    cfg["train_semantic_path"] = str(s1_dir / "6-name2semantic.tsv")
    cfg["train_phoneme_path"] = str(s1_dir / "2-name2text.txt")
    cfg["output_dir"] = str(s1_dir / "logs_s1_v2ProPlus")

    tmp_cfg = s1_dir / "s1_run_config.yaml"
    tmp_cfg.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

    env = base_env()
    env["_CUDA_VISIBLE_DEVICES"] = "0"
    env["hz"] = "25hz"
    return run_cmd(
        [str(PYTHON_EXE), "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(tmp_cfg)],
        env=env,
        label=f"s1_train ({exp_name}, bs={batch_size}, ep={epochs})",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--list_path", required=True)
    parser.add_argument("--s2_batch", type=int, default=6)
    parser.add_argument("--s2_epochs", type=int, default=8)
    parser.add_argument("--s2_save_every", type=int, default=4)
    parser.add_argument("--s1_batch", type=int, default=6)
    parser.add_argument("--s1_epochs", type=int, default=12)
    parser.add_argument("--s1_save_every", type=int, default=4)
    parser.add_argument("--n_parts", type=int, default=2)
    args = parser.parse_args()

    ensure_users_pth()
    t_total = time.time()
    timings = run_prepare(args.exp_name, Path(args.list_path), n_parts=args.n_parts)
    timings["s2_train"] = run_s2(args.exp_name, args.s2_batch, args.s2_epochs, args.s2_save_every)
    timings["s1_train"] = run_s1(args.exp_name, args.s1_batch, args.s1_epochs, args.s1_save_every)
    timings["total_sec"] = time.time() - t_total
    timings["total_min"] = timings["total_sec"] / 60.0

    out_json = Path(r"e:\voiceSynth\data") / f"benchmark_{args.exp_name}.json"
    out_json.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    print("\n=== BENCHMARK SUMMARY ===")
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
