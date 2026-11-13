"""Generate long multi-sentence test sequences across saved ichika_4h checkpoints."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_DIR = Path(r"e:\voiceSynth\GPT-SoVITS")
OUT_DIR = Path(r"e:\voiceSynth\outputs\ichika_4h_samples")

REF_WAV = r"e:\voiceSynth\references\ichika_ref.wav"
REF_TEXT = "あの日の咲希、ずっとテンション高かったよね"

LONG_TEXTS = {
    "long_story": (
        "放課後の音楽室って、夕方の光が差し込んでいてすごく落ち着くんだよね。"
        "咲希がキーボードを弾いて、穂波がドラムのリズムを合わせて、志歩がベースで全体を支えてくれる。"
        "みんなで音を鳴らしていると、どんなに迷っていたことでも、前を向いて歩き出せる気がするんだ。"
        "これからも、私達らしい音楽をたくさん届けていけたらいいな。"
    ),
    "expressive_dialogue": (
        "あっ、みんな！次のライブのセットリスト、そろそろ決めないとだよね？"
        "えっと、一曲目はやっぱりアップテンポな曲で勢いをつけたいなって思ってるんだけど、どうかな。"
        "もし不安なところがあったら、今日も居残りで練習しよう！絶対に最高のステージにしようね！"
    ),
}


from synthesize import split_japanese_sentences


def main() -> None:
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

    i18n = I18nAuto(language="en_US")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check available checkpoints
    sovits_dir = REPO_DIR / "SoVITS_weights_v2ProPlus"
    gpt_dir = REPO_DIR / "GPT_weights_v2ProPlus"
    sovits_ckpts = sorted(sovits_dir.glob("ichika_4h_*.pth"))
    gpt_ckpts = sorted(gpt_dir.glob("ichika_4h-*.ckpt"))
    print("SoVITS checkpoints:", [p.name for p in sovits_ckpts])
    print("GPT checkpoints:", [p.name for p in gpt_ckpts])

    # Evaluate final SoVITS (e8) paired with GPT final e12 using proper per-sentence splitting
    combos = [
        (sovits_dir / "ichika_4h_e8_s3608.pth", gpt_dir / "ichika_4h-e12.ckpt"),
    ]
    combos = [(s_p, g_p) for s_p, g_p in combos if s_p.exists() and g_p.exists()]
    if not combos and sovits_ckpts and gpt_ckpts:
        combos = [(sovits_ckpts[-1], gpt_ckpts[-1])]

    for s_p, g_p in combos:
        iw.change_gpt_weights(gpt_path=str(g_p))
        for _ in iw.change_sovits_weights(sovits_path=str(s_p)):
            pass
        for name, text in LONG_TEXTS.items():
            t0 = time.time()
            chunked_text = split_japanese_sentences(text)
            gen = iw.get_tts_wav(
                ref_wav_path=REF_WAV,
                prompt_text=REF_TEXT,
                prompt_language=i18n("日文"),
                text=chunked_text,
                text_language=i18n("日文"),
                how_to_cut=i18n("不切"),
                top_k=15,
                top_p=0.6,
                temperature=0.6,
                speed=1.0,
                pause_second=0.45,
            )
            results = list(gen)
            sr, audio = results[-1]
            dt = time.time() - t0
            dur = len(audio) / sr
            rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
            peak = float(np.max(np.abs(audio)))
            out_wav = OUT_DIR / f"{s_p.stem}__{g_p.stem}__{name}__dry_ref.wav"
            sf.write(str(out_wav), audio, sr)
            print(f"OK {out_wav.name}: dur={dur:.2f}s rtf={dt/dur:.3f} rms={rms:.4f} peak={peak:.4f}")


if __name__ == "__main__":
    main()
