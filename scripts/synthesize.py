"""Clean CLI synthesis & checkpoint comparison script for fine-tuned GPT-SoVITS v2ProPlus."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf

REPO_DIR = Path(r"e:\voiceSynth\GPT-SoVITS")


def init_gptsovits():
    os.chdir(REPO_DIR)
    for p in [
        str(REPO_DIR),
        str(REPO_DIR / "GPT_SoVITS"),
        str(REPO_DIR / "GPT_SoVITS" / "BigVGAN"),
        str(REPO_DIR / "tools"),
        str(REPO_DIR / "tools" / "asr"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)

    os.environ["version"] = "v2ProPlus"
    os.environ["is_half"] = "True"
    os.environ["cnhubert_base_path"] = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    os.environ["bert_path"] = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

    from tools.i18n.i18n import I18nAuto
    import GPT_SoVITS.inference_webui as iw

    return iw, I18nAuto(language="en_US")


def split_japanese_sentences(text: str, min_chars: int = 8) -> str:
    """Split Japanese text on terminal sentence punctuation into natural 3-8s chunks."""
    import re

    parts = [p.strip() for p in re.split(r"(?<=[。！？!?])", text.strip()) if p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] += p
        else:
            merged.append(p)
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        merged[-2] += merged[-1]
        merged.pop()
    return "\n".join(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Japanese speech with GPT-SoVITS v2ProPlus")
    parser.add_argument("--gpt_model", required=True, help="Path to fine-tuned GPT .ckpt")
    parser.add_argument("--sovits_model", required=True, help="Path to fine-tuned SoVITS .pth")
    parser.add_argument("--ref_audio", required=True, help="Reference WAV path (3-10s clean line)")
    parser.add_argument("--ref_text", required=True, help="Transcript of reference WAV")
    parser.add_argument("--text", required=True, help="Target Japanese text to synthesize")
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pause_second", type=float, default=0.45)
    parser.add_argument("--cut", default="sentence", help="Split mode: sentence (default), 50, 4, punc, none")
    args = parser.parse_args()

    iw, i18n = init_gptsovits()
    iw.change_gpt_weights(gpt_path=args.gpt_model)
    # change_sovits_weights is a generator
    for _ in iw.change_sovits_weights(sovits_path=args.sovits_model):
        pass

    target_text = args.text
    if args.cut == "sentence":
        target_text = split_japanese_sentences(target_text)
        how_to_cut = i18n("不切")
    else:
        cut_map = {
            "none": i18n("不切"),
            "不切": i18n("不切"),
            "50": i18n("凑50字一切"),
            "凑50字一切": i18n("凑50字一切"),
            "4": i18n("凑四句一切"),
            "凑四句一切": i18n("凑四句一切"),
            "punc": i18n("按标点符号切"),
            "按标点符号切": i18n("按标点符号切"),
        }
        how_to_cut = cut_map.get(args.cut, i18n("凑50字一切"))

    gen = iw.get_tts_wav(
        ref_wav_path=args.ref_audio,
        prompt_text=args.ref_text,
        prompt_language=i18n("日文"),
        text=target_text,
        text_language=i18n("日文"),
        how_to_cut=how_to_cut,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        speed=args.speed,
        pause_second=args.pause_second,
    )
    results = list(gen)
    if not results:
        raise RuntimeError("Synthesis produced no audio")
    sr, audio = results[-1]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr)
    print(f"Saved {out_path} ({len(audio) / sr:.2f}s @ {sr} Hz)")


if __name__ == "__main__":
    main()
