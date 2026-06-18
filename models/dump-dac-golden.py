#!/usr/bin/env python3
"""Golden dump for the DAC decoder port.

Reads RAW C++-style codes [T,9] (+ optional .eos.npy), reproduces decode-codes.py exactly
(shear_up pad=1025 -> truncate at eos -> clamp<=1023 -> from_codes -> decoder), and dumps
fp32 intermediates + the final audio so the C++ dac-cli can be diffed stage by stage.

Runs on CPU (matches the C++ CPU path apples-to-apples).

    PYTHONPATH= /data/home/sofian/ZONOS2/.venv/bin/python models/dump-dac-golden.py \
        out/clone_tts.npy out/dac_golden
"""
import os
import sys

import numpy as np
import torch

AUDIO_PAD_ID = 1025


def main():
    codes_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    from zonos2.tokenizer.vocoder import shear_up
    import dac as dac_module

    model = dac_module.DAC.load(
        dac_module.utils.download(model_type="44khz")
    ).eval().to("cpu")

    raw = np.load(codes_path).round().astype(np.int64)          # [T, 9]
    codes = torch.tensor(raw, device="cpu")
    eos_path = codes_path + ".eos.npy"
    eos_frame = None
    if os.path.exists(eos_path):
        ev = int(np.load(eos_path)[0])
        eos_frame = ev if ev >= 0 else None

    codes = shear_up(codes, AUDIO_PAD_ID)
    if eos_frame is not None:
        codes = codes[: max(0, eos_frame)]
    print("codes after shear/trunc:", tuple(codes.shape), "eos:", eos_frame)
    codes = torch.clamp(codes, max=1023)
    np.save(os.path.join(out_dir, "codes_proc.npy"), codes.numpy().astype(np.int32))  # [T,9]

    # DAC expects (batch, codebooks, seq)
    codes_in = codes.permute(1, 0).unsqueeze(0)                  # [1, 9, T]

    # capture decoder stage outputs
    stages = {}
    dec = model.decoder.model  # Sequential: 0 conv_in, 1..4 blocks, 5 snake, 6 conv, 7 tanh
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            stages[i] = out.detach().cpu().float().numpy()
        return hook

    for i in range(len(dec)):
        handles.append(dec[i].register_forward_hook(mk(i)))

    with torch.no_grad(), torch.inference_mode():
        z = model.quantizer.from_codes(codes_in)[0]             # [1, latent, T]
        audio = model.decode(z).float().squeeze(1).squeeze(0)   # [samples]

    for h in handles:
        h.remove()

    # z is [1, latent, T] -> store [latent, T]
    np.save(os.path.join(out_dir, "z.npy"), z.squeeze(0).cpu().numpy())
    # stage[i] is [1, C, L] -> store [C, L]
    names = {0: "conv_in", 1: "block0", 2: "block1", 3: "block2", 4: "block3",
             5: "snake_out", 6: "conv_out", 7: "tanh"}
    for i, nm in names.items():
        np.save(os.path.join(out_dir, f"{nm}.npy"), stages[i].squeeze(0))
        print(f"  {nm:9s} {stages[i].shape}")
    np.save(os.path.join(out_dir, "audio.npy"), audio.cpu().numpy())
    print("audio:", audio.shape, "rms", float(audio.pow(2).mean().sqrt()),
          "peak", float(audio.abs().max()))
    print("wrote golden ->", out_dir)


if __name__ == "__main__":
    main()
