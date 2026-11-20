"""Clean CLI synthesis & checkpoint comparison script for fine-tuned GPT-SoVITS v2ProPlus."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf

REPO_DIR = Path(r"e:\voiceSynth\GPT-SoVITS")


def init_gptsovits(device_mode: str = "auto"):
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
    if device_mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["_CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["is_half"] = "False"
    else:
        os.environ.setdefault("is_half", "True")
    os.environ["cnhubert_base_path"] = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    os.environ["bert_path"] = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

    from tools.i18n.i18n import I18nAuto
    import GPT_SoVITS.inference_webui as iw

    return iw, I18nAuto(language="en_US")


def split_sentences(text: str, min_chars: int = 8) -> str:
    """Split Japanese/English text on terminal sentence punctuation into natural chunks."""
    import re

    parts = [p.strip() for p in re.split(r"(?<=[。！？!?])|(?<!\d)\.(?!\d)\s+", text.strip()) if p.strip()]
    # Ensure split English periods aren't dropped before chunking
    restored: list[str] = []
    for p in parts:
        if p and p[-1] not in "。！？!?." and any("a" <= c.lower() <= "z" for c in p):
            p = p + "."
        restored.append(p)
    merged: list[str] = []
    for p in restored:
        if merged and len(merged[-1]) < min_chars:
            sep = " " if merged[-1][-1].isascii() and p[0].isascii() else ""
            merged[-1] += sep + p
        else:
            merged.append(p)
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        sep = " " if merged[-2][-1].isascii() and merged[-1][0].isascii() else ""
        merged[-2] += sep + merged[-1]
        merged.pop()
    return "\n".join(merged)


split_japanese_sentences = split_sentences


LANG_MAP = {
    "ja": "日文",
    "en": "英文",
    "ja_en": "日英混合",
    "zh": "中文",
    "zh_en": "中英混合",
    "ko": "韩文",
    "ko_en": "韩英混合",
    "yue": "粤语",
    "auto": "多语种混合",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize multilingual speech with GPT-SoVITS v2ProPlus")
    parser.add_argument("--gpt_model", required=True, help="Path to fine-tuned GPT .ckpt")
    parser.add_argument("--sovits_model", required=True, help="Path to fine-tuned SoVITS .pth")
    parser.add_argument("--ref_audio", required=True, help="Reference WAV path (3-10s clean line)")
    parser.add_argument("--ref_text", required=True, help="Transcript of reference WAV")
    parser.add_argument("--text", required=True, help="Target text to synthesize")
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--lang", default="ja", choices=sorted(LANG_MAP.keys()), help="Target text language")
    parser.add_argument("--prompt_lang", default="ja", choices=sorted(LANG_MAP.keys()), help="Reference prompt language")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pause_second", type=float, default=0.45)
    parser.add_argument("--cut", default="sentence", help="Split mode: sentence (default), 50, 4, punc, none")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu"], help="Execution device: auto (GPU if available) or cpu")
    args = parser.parse_args()

    iw, i18n = init_gptsovits(args.device)
    iw.change_gpt_weights(gpt_path=args.gpt_model)
    # change_sovits_weights is a generator
    for _ in iw.change_sovits_weights(sovits_path=args.sovits_model):
        pass

    target_text = args.text
    if args.cut == "sentence":
        target_text = split_sentences(target_text)
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
        prompt_language=i18n(LANG_MAP[args.prompt_lang]),
        text=target_text,
        text_language=i18n(LANG_MAP[args.lang]),
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
