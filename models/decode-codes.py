#!/usr/bin/env python3
"""Decode C++-generated audio codes [T, 9] to a WAV via shear_up + DAC (mirrors TTSLLM).

    CUDA_VISIBLE_DEVICES=0 /data/home/sofian/ZONOS2/.venv/bin/python \
        models/decode-codes.py out/codes_sampled.npy out/hello.wav
"""
import os
import sys

import wave

import numpy as np
import torch

AUDIO_PAD_ID = 1025


def write_wav(path, mono_f32, sr=44100):
    pcm = np.clip(mono_f32.numpy(), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


def main():
    codes_path = sys.argv[1]
    out_wav = sys.argv[2]

    from zonos2.tokenizer.vocoder import TTSVocoderManager, shear_up

    codes_np = np.load(codes_path)
    codes = torch.tensor(codes_np.round().astype(np.int64), device="cuda")  # [T, 9]
    print("codes:", tuple(codes.shape))

    eos_path = codes_path + ".eos.npy"
    eos_frame = None
    if os.path.exists(eos_path):
        ev = int(np.load(eos_path)[0])
        eos_frame = ev if ev >= 0 else None

    codes = shear_up(codes, AUDIO_PAD_ID)
    if eos_frame is not None:
        codes = codes[: max(0, eos_frame)]
        print("truncated to eos_frame:", eos_frame, "->", tuple(codes.shape))
    if codes.numel() == 0:
        print("no frames to decode")
        return

    voc = TTSVocoderManager(n_codebooks=9, audio_pad_id=AUDIO_PAD_ID)
    audio = voc.decode_all(codes.unsqueeze(0), apply_shear_up=False)  # [1, samples]
    wav = audio[0].detach().float().cpu()
    print(f"audio: {wav.shape[-1]} samples = {wav.shape[-1] / 44100:.2f}s, "
          f"rms={wav.pow(2).mean().sqrt():.4f}, peak={wav.abs().max():.4f}")
    write_wav(out_wav, wav, 44100)
    print("wrote", out_wav)


if __name__ == "__main__":
    main()
