#!/usr/bin/env python3
"""Compare two directories of .npy tensors (golden vs C++) in a stable order.

    python models/compare.py out/golden out/cpp [--only logits,emb_norm,...]

Reports max-abs, max-rel, cosine, and (for logits) per-codebook argmax agreement.
Walk the printed order top-to-bottom; the first FAIL localizes the bug.
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np


def _key(name):
    # natural sort so attn_out_2 < attn_out_10, and layer order is sensible
    m = re.match(r"(.*?)(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, -1)


def metrics(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.shape != b.shape:
        return None
    d = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-6)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return dict(maxabs=float(d.max()), maxrel=float((d / denom).max()),
                rmse=float(np.sqrt(np.mean(d * d))), cos=cos)


def argmax_agree(g, c):
    # logits [n, n_cb, vocab] -> fraction of (token,codebook) with same argmax
    if g.ndim != 3 or g.shape != c.shape:
        return None
    ag = g.argmax(-1); ac = c.argmax(-1)
    return float((ag == ac).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("cpp")
    ap.add_argument("--only", default=None, help="comma list of names to restrict to")
    ap.add_argument("--cos", type=float, default=0.999, help="cosine pass threshold")
    args = ap.parse_args()

    gfiles = {f[:-4] for f in os.listdir(args.golden) if f.endswith(".npy")}
    cfiles = {f[:-4] for f in os.listdir(args.cpp) if f.endswith(".npy")}
    names = sorted(gfiles & cfiles, key=_key)
    if args.only:
        want = set(args.only.split(","))
        names = [n for n in names if n in want]

    only_g = sorted(gfiles - cfiles, key=_key)
    if only_g and not args.only:
        print(f"(only in golden, skipped: {len(only_g)} e.g. {only_g[:6]})")

    print(f"{'tensor':<22}{'shape':<18}{'maxabs':>10}{'maxrel':>10}{'cos':>11}  status")
    n_fail = 0
    for n in names:
        g = np.load(os.path.join(args.golden, n + ".npy"))
        c = np.load(os.path.join(args.cpp, n + ".npy"))
        m = metrics(g, c)
        if m is None:
            print(f"{n:<22}{str(g.shape)+' vs '+str(c.shape):<18}{'':>31}  SHAPE-MISMATCH")
            n_fail += 1
            continue
        ok = m["cos"] >= args.cos
        extra = ""
        if n in ("logits", "logits_in", "mout"):
            aa = argmax_agree(g, c)
            if aa is not None:
                # last position generates frame 0; report it separately (it is what matters)
                last = float((g[-1].argmax(-1) == c[-1].argmax(-1)).mean()) if g.ndim == 3 else aa
                extra = f"  argmax={aa*100:.2f}% last={last*100:.1f}%"
                ok = ok and aa > 0.97  # near-tie flips at irrelevant prompt positions are expected (bf16 ref)
        status = "ok" if ok else "FAIL"
        n_fail += (not ok)
        print(f"{n:<22}{str(g.shape):<18}{m['maxabs']:>10.4g}{m['maxrel']:>10.4g}{m['cos']:>11.6f}  {status}{extra}")

    print(f"\n{len(names)} compared, {n_fail} fail(s)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
