#!/usr/bin/env python3
"""Extract [2048] speaker embeddings from reference audio using the release encoder
(Qwen3SpeakerEmbedding), mirroring api_server._compute_speaker_embedding_from_waveform.
Saves one <out_dir>/<basename>.npy per input, ready for `zonos2-cli --speaker`.

torchaudio 2.9 needs torchcodec to load, so audio is transcoded mp3/etc -> wav via ffmpeg.

    CUDA_VISIBLE_DEVICES=1 .../python models/embed-voice.py out/voices default_voices/*.mp3
"""
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import torch


def load_mono(path):
    wav_path = tempfile.mktemp(suffix=".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", "44100", wav_path], check=True)
    w = wave.open(wav_path, "rb")
    sr, n = w.getframerate(), w.getnframes()
    raw = w.readframes(n)
    w.close()
    os.unlink(wav_path)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0), sr  # [1, N]


def main():
    out_dir, srcs = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

    enc = Qwen3SpeakerEmbedding(device="cuda")
    for src in srcs:
        wav, sr = load_mono(src)
        with torch.inference_mode():
            out = enc(wav, sr)
        emb = out.squeeze(0).to(torch.float32).cpu().reshape(-1).numpy()
        assert emb.shape[0] == 2048, f"{src}: got {emb.shape}"
        name = os.path.splitext(os.path.basename(src))[0]
        np.save(os.path.join(out_dir, name + ".npy"), emb)
        print(f"{src}: wav {tuple(wav.shape)}@{sr}Hz -> {name}.npy "
              f"[2048] norm={np.linalg.norm(emb):.2f}", flush=True)


if __name__ == "__main__":
    main()
