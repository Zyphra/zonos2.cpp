#!/usr/bin/env python3
"""Convert the ECAPA-TDNN speaker encoder to GGUF for the standalone C++ port.

The HF repo is named marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B but the model is
actually a ~6M-param ECAPA-TDNN (architectures=["EcapaTdnnSpeakerEncoder"]), not a
Qwen3 transformer. It maps a 128-bin log-mel spectrogram -> a 2048-d x-vector.

Conv1d weights are PyTorch [OC, IC, K]; we store ggml ne=[IC, OC, K] (numpy [K,OC,IC]),
i.e. K kernel slices each an [IC,OC] matmul -- the C++ side runs a conv as a sum of K
dilated 1x1 matmuls over the channel axis (k=1 convs are a single matmul). Biases are
1-D f32. We also stash the slaney mel filterbank and the analysis window so the C++
side can build log-mel from a 24 kHz mono waveform (STFT via DFT matmul). Conv weights
default to f16 (lossless from the bf16 originals, ~12 MB gguf); biases + mel_fb + window
stay f32. Pass --outtype f32 for a ~24 MB exact file.

  /data/home/sofian/ZONOS2/.venv/bin/python models/convert-spk-encoder-to-gguf.py \
      <hf_snapshot_dir> -o out/spk-encoder.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_pydeps"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gguf import GGUFWriter  # noqa: E402

ARCH = "ecapa-tdnn"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="HF snapshot dir with config.json + model.safetensors")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--outtype", choices=["f16", "f32"], default="f16",
                    help="conv weight dtype (f16 default: lossless from bf16, ~halves the file)")
    args = ap.parse_args()

    with open(os.path.join(args.snapshot, "config.json")) as f:
        cfg = json.load(f)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(args.snapshot, "model.safetensors"))
    log("config:", {k: cfg[k] for k in ("mel_dim", "enc_dim", "enc_channels",
        "enc_kernel_sizes", "enc_dilations", "enc_attention_channels",
        "enc_res2net_scale", "enc_se_channels")})

    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "ECAPA-TDNN-SpeakerEncoder")
    w.add_uint32(f"{ARCH}.mel_dim", cfg["mel_dim"])
    w.add_uint32(f"{ARCH}.enc_dim", cfg["enc_dim"])
    w.add_array(f"{ARCH}.enc_channels", cfg["enc_channels"])
    w.add_array(f"{ARCH}.enc_kernel_sizes", cfg["enc_kernel_sizes"])
    w.add_array(f"{ARCH}.enc_dilations", cfg["enc_dilations"])
    w.add_uint32(f"{ARCH}.attention_channels", cfg["enc_attention_channels"])
    w.add_uint32(f"{ARCH}.res2net_scale", cfg["enc_res2net_scale"])
    w.add_uint32(f"{ARCH}.se_channels", cfg["enc_se_channels"])
    w.add_uint32(f"{ARCH}.sample_rate", cfg.get("sample_rate", 24000))

    # --- mel frontend constants (mirror speaker_cloning.Qwen3SpeakerEmbedding) ---
    N_FFT, HOP, WIN, N_MELS, FMIN, FMAX, SR = 1024, 256, 1024, 128, 0.0, 12000.0, 24000
    import torchaudio
    mt = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        f_min=FMIN, f_max=FMAX, n_mels=N_MELS, power=1.0, center=False,
        norm="slaney", mel_scale="slaney")
    fb = mt.mel_scale.fb.numpy().astype(np.float32)          # [n_freq=513, n_mels=128]
    window = mt.spectrogram.window.numpy().astype(np.float32)  # [win=1024]
    w.add_uint32(f"{ARCH}.n_fft", N_FFT)
    w.add_uint32(f"{ARCH}.hop_length", HOP)
    w.add_uint32(f"{ARCH}.win_length", WIN)
    w.add_float32(f"{ARCH}.mel_fmin", FMIN)
    w.add_float32(f"{ARCH}.mel_fmax", FMAX)
    # store ggml ne=[n_freq=513, n_mels=128] -> numpy [128, 513]; window 1-D
    w.add_tensor("mel_fb", np.ascontiguousarray(fb.T))   # [128, 513] -> ne=[513,128]
    w.add_tensor("mel_window", np.ascontiguousarray(window))
    log(f"mel_fb {fb.T.shape} window {window.shape}")

    added = 0

    def add_conv(name, t):
        nonlocal added
        a = t.detach().cpu().float().numpy()  # PyTorch [OC, IC, K]
        assert a.ndim == 3, f"{name}: expected 3-D conv weight, got {a.shape}"
        a = np.ascontiguousarray(np.transpose(a, (2, 0, 1)))  # -> [K, OC, IC] => ggml ne=[IC,OC,K]
        if args.outtype == "f16":   # bf16 originals fit f16 losslessly; mul_mat takes f16 w * f32 act
            mx = float(np.abs(a).max())
            if mx >= 65504.0:
                raise SystemExit(f"f16 overflow in {name}: max|x|={mx}")
            a = a.astype(np.float16)
        w.add_tensor(name, a)
        added += 1

    def add_vec(name, t):
        nonlocal added
        w.add_tensor(name, np.ascontiguousarray(t.detach().cpu().float().numpy()))
        added += 1

    for k in sorted(sd.keys()):
        t = sd[k]
        if k.endswith(".weight"):
            add_conv(k, t)
        elif k.endswith(".bias"):
            add_vec(k, t)
        else:
            raise SystemExit(f"unexpected tensor {k} {tuple(t.shape)}")

    log(f"mapped {added} model tensors (+2 mel) -> {args.out}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    log("done.")


if __name__ == "__main__":
    main()
