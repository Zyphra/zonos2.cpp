#!/usr/bin/env python3
"""Stage-B check: reimplement the ECAPA-TDNN forward in numpy from the GGUF weights
+ golden mel, and compare every stage to the golden dump. Validates the conv weight
layout (ggml ne=[IC,OC,K]) and my architecture understanding before the C++ port.

  .../python models/verify-spk-numpy.py out/spk-encoder.gguf out/spk_golden
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_pydeps"))
import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def load_gguf(path):
    # GGUFReader.data is numpy-order (reversed ggml ne): a conv stored ggml ne=[IC,OC,K]
    # comes back as numpy [K,OC,IC]. Use it as-is (do NOT reshape to ggml order).
    r = GGUFReader(path)
    return {t.name: np.array(t.data) for t in r.tensors}


def main():
    gguf_path, gold = sys.argv[1], sys.argv[2]
    W = load_gguf(gguf_path)
    g = lambda n: np.load(os.path.join(gold, n + ".npy"))

    def conv(x, name, dil, k):
        """x [IC,T] -> [OC,T]. weight numpy [K,OC,IC] (reversed ggml ne), reflect 'same' pad."""
        w = W[name + ".weight"]            # [K, OC, IC]
        b = W[name + ".bias"]              # [OC]
        K, OC, IC = w.shape
        assert K == k, f"{name}: K {K}!={k}"
        p = dil * (k - 1) // 2
        if p > 0:
            left = x[:, 1:p + 1][:, ::-1]            # reflect (mirror around col 0)
            right = x[:, -p - 1:-1][:, ::-1]
            xp = np.concatenate([left, x, right], axis=1)
        else:
            xp = x
        T = x.shape[1]
        out = np.zeros((OC, T), np.float32)
        for j in range(k):
            out += w[j] @ xp[:, j * dil: j * dil + T]   # [OC,IC]@[IC,T] = [OC,T]
        return out + b[:, None]

    def tdnn(x, name, dil=1, k=1):
        return np.maximum(conv(x, name, dil, k), 0.0)  # conv + ReLU

    def res2net(x, prefix, dil, scale=8):
        chunks = np.split(x, scale, axis=0)
        outs = []
        prev = None
        for i, c in enumerate(chunks):
            if i == 0:
                op = c
            elif i == 1:
                op = tdnn(c, f"{prefix}.blocks.0.conv", dil, 3)
            else:
                op = tdnn(c + prev, f"{prefix}.blocks.{i-1}.conv", dil, 3)
            prev = op
            outs.append(op)
        return np.concatenate(outs, axis=0)

    def se(x, prefix):
        m = x.mean(axis=1, keepdims=True)             # [IC,1]
        m = np.maximum(conv(m, f"{prefix}.conv1", 1, 1), 0.0)
        m = 1.0 / (1.0 + np.exp(-conv(m, f"{prefix}.conv2", 1, 1)))
        return x * m

    def se_res2net(x, i, dil):
        p = f"blocks.{i}"
        r = x
        h = tdnn(x, f"{p}.tdnn1.conv", 1, 1)
        h = res2net(h, f"{p}.res2net_block", dil)
        h = tdnn(h, f"{p}.tdnn2.conv", 1, 1)
        h = se(h, f"{p}.se_block")
        return h + r

    mel = g("mel")          # [T,128]
    x = np.ascontiguousarray(mel.T).astype(np.float32)  # [128,T]

    # block0: TDNN k=5 d=1
    h = tdnn(x, "blocks.0.conv", 1, 5)
    print(f"block0   cos={cos(h, g('block0')):.6f}")
    feats = []
    for i, dil in [(1, 2), (2, 3), (3, 4)]:
        h = se_res2net(h, i, dil)
        print(f"block{i}   cos={cos(h, g(f'block{i}')):.6f}")
        feats.append(h)
    h = np.concatenate(feats, axis=0)        # [1536,T]
    h = tdnn(h, "mfa.conv", 1, 1)
    print(f"mfa      cos={cos(h, g('mfa')):.6f}")

    # ASP
    T = h.shape[1]
    mean_g = h.mean(axis=1)
    std_g = np.sqrt(np.clip(((h - mean_g[:, None]) ** 2).mean(axis=1), 1e-12, None))
    ctx = np.concatenate([h, np.repeat(mean_g[:, None], T, 1), np.repeat(std_g[:, None], T, 1)], axis=0)
    a = np.maximum(conv(ctx, "asp.tdnn.conv", 1, 1), 0.0)
    a = np.tanh(a)
    a = conv(a, "asp.conv", 1, 1)            # [1536,T]
    a = a - a.max(axis=1, keepdims=True)
    aw = np.exp(a); aw /= aw.sum(axis=1, keepdims=True)
    mean = (aw * h).sum(axis=1)
    std = np.sqrt(np.clip((aw * (h - mean[:, None]) ** 2).sum(axis=1), 1e-12, None))
    pooled = np.concatenate([mean, std])     # [3072]
    print(f"asp      cos={cos(pooled, g('asp').ravel()):.6f}")

    emb = conv(pooled[:, None], "fc", 1, 1).ravel()  # [2048]
    print(f"emb      cos={cos(emb, g('emb')):.6f}  |emb|={np.linalg.norm(emb):.4f} golden={np.linalg.norm(g('emb')):.4f}")


if __name__ == "__main__":
    main()
