#!/usr/bin/env python3
"""Golden dump for the ECAPA-TDNN speaker encoder C++ port.

Decodes a reference audio to 24 kHz mono (ffmpeg), runs the real
Qwen3SpeakerEmbedding (ECAPA-TDNN) on CPU, and dumps fp32 .npy for:
  wav24k [N], mel [T,128] (model input), per-block activations [C,T],
  asp [3072], fc/emb [2048].
Activations are captured by monkeypatching submodule .forward (saved as [C,T]).

  CUDA_VISIBLE_DEVICES= .../python models/dump-spk-golden.py <audio> -o out/spk_golden
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch


def load_wav_24k_mono(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "24000",
         "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").copy()  # [N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="out/spk_golden")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

    wav = load_wav_24k_mono(args.audio)
    np.save(os.path.join(args.out, "wav24k.npy"), wav.astype(np.float32))
    print(f"wav24k: {wav.shape} @24kHz ({wav.shape[0]/24000:.2f}s)", flush=True)

    enc = Qwen3SpeakerEmbedding(device="cpu")

    saved = {}

    def cap(name, t):
        saved[name] = t.detach().cpu().float().squeeze(0).numpy()

    # patch block/mfa/asp/fc forwards to snapshot outputs ([C,T] or vec)
    m = enc.model
    for i, blk in enumerate(m.blocks):
        orig = blk.forward
        def mk(o, nm):
            def f(*a, **k):
                out = o(*a, **k); cap(nm, out); return out
            return f
        blk.forward = mk(orig, f"block{i}")
    for nm in ("mfa", "asp", "fc"):
        sub = getattr(m, nm); o = sub.forward
        def mk2(o, nm):
            def f(*a, **k):
                out = o(*a, **k); cap(nm, out); return out
            return f
        sub.forward = mk2(o, nm)

    wt = torch.from_numpy(wav).unsqueeze(0)  # [1, N]
    with torch.inference_mode():
        # mirror Qwen3SpeakerEmbedding.forward, but capture mel (sr=24k -> no resample)
        w24 = enc.prepare_input(wt, 24000)
        mel = enc._make_mel(w24)              # [1, T, 128]
        cap("mel", mel)
        out = enc.model(input_values=mel).last_hidden_state.to(torch.float32)
    cap("emb", out)

    for k, v in saved.items():
        np.save(os.path.join(args.out, k + ".npy"), np.ascontiguousarray(v))
        print(f"  {k:8} {tuple(v.shape)}", flush=True)
    e = saved["emb"]
    print(f"emb norm={np.linalg.norm(e):.4f}  (fc==emb: {np.allclose(saved['fc'], e)})", flush=True)


if __name__ == "__main__":
    main()
