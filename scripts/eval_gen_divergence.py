#!/usr/bin/env python3
"""Generation-based quant evaluation: greedy-decode held-out prompts with two models and
compare the produced code sequences. Greedy removes sampling noise, so any divergence is the
quantization itself. This closes the autoregressive feedback loop that teacher-forced KLD
cannot see — the metric that mattered for the q2 audio-quality regression.

Usage: eval_gen_divergence.py <ref.gguf> <test.gguf> [--max N] [--gpu G]
"""
import subprocess, sys, tempfile, os, glob
import numpy as np

CLI = "./build-cuda/zonos2-cli"
# Held-out prompts — deliberately NOT in the calibration corpus, to avoid optimistic bias.
PROMPTS = [
    "The lighthouse keeper climbed the spiral staircase as the storm battered the rocky coast below.",
    "Quantum computers exploit superposition and entanglement to solve certain problems much faster.",
    "Good evening, and welcome back to the show; tonight we have a truly remarkable guest for you.",
    "She folded the letter carefully, placed it in the drawer, and never spoke of it again.",
    "The marathon route winds through the old town, past the cathedral, and finishes at the bridge.",
    "Frankly, I did not expect the committee to approve the proposal on its very first reading.",
]

def gen(model, text, max_frames, gpu, tmp):
    out = os.path.join(tmp, "c.npy")
    for f in glob.glob(os.path.join(tmp, "c.npy*")):
        os.remove(f)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    r = subprocess.run([CLI, model, "--tts", text, out, "--greedy", "--max", str(max_frames), "--gpu"],
                       env=env, capture_output=True, text=True)
    if not os.path.exists(out):
        sys.stderr.write(r.stderr[-2000:]); raise RuntimeError(f"gen failed for {model}")
    return np.load(out).astype(np.int64)  # [n_frames, n_codebooks]

def main():
    ref, test = sys.argv[1], sys.argv[2]
    mx = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 300
    gpu = sys.argv[sys.argv.index("--gpu") + 1] if "--gpu" in sys.argv else "0"
    print(f"ref = {ref}\ntest= {test}\nmax = {mx}, gpu = {gpu}\n")
    print(f"{'prompt':6s} {'frames(r/t)':>12s} {'1st-div':>8s} {'frame-match%':>12s} {'code-match%':>11s}")
    fm_all, cm_all, div_all = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, p in enumerate(PROMPTS):
            a = gen(ref, p, mx, gpu, tmp)
            b = gen(test, p, mx, gpu, tmp)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            frame_eq = np.all(a == b, axis=1)            # whole frame identical
            code_eq = (a == b)                           # per-codebook
            first_div = int(np.argmin(frame_eq)) if not frame_eq.all() else n
            fm = 100.0 * frame_eq.mean()
            cm = 100.0 * code_eq.mean()
            fm_all.append(fm); cm_all.append(cm); div_all.append(first_div)
            print(f"  p{i:<3d} {len(a):5d}/{len(b):<5d} {first_div:8d} {fm:12.2f} {cm:11.2f}")
    print(f"\nmean: 1st-div={np.mean(div_all):.0f} frame-match={np.mean(fm_all):.2f}% "
          f"code-match={np.mean(cm_all):.2f}%")
    print("(frame-match% = fraction of frames where ALL codebooks match the f16 reference; "
          "higher = closer to f16. A good quant stays high and diverges late.)")

if __name__ == "__main__":
    main()
