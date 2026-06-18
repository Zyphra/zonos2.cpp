#!/usr/bin/env python3
"""Convert the Zyphra/ZONOS2 checkpoint (model.pth + params.json) to GGUF for the
standalone zonos2.cpp ggml backbone.

Default outtype is f16: bf16->f16 is effectively lossless for these (O(1)) weights
(bf16's 7 mantissa bits fit in f16's 10; values are well within f16's +-65504 range),
so one f16 file serves both CPU bring-up and CUDA and still matches the fp32 reference.
1-D tensors (norms, biases, temp, eda-scale) are always stored f32. In a q8_0 file the
quant-sensitive matrices -- the embedding tables, the output head, and the router weights
-- are bumped up to f16; everything else is Q8_0 (see pick_qtype).

Usage:
  /data/home/sofian/ZONOS2/.venv/bin/python models/convert-zonos2-to-gguf.py \
      <hf_snapshot_dir> --outtype f16 -o out/zonos2-f16.gguf
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
from gguf import GGUFWriter, GGMLQuantizationType  # noqa: E402
import gguf.quants as gquants  # noqa: E402

ARCH = "zonos2"


def log(*a):
    print(*a, flush=True)


def load_state(snap):
    with open(os.path.join(snap, "params.json")) as f:
        params = json.load(f)
    sd = torch.load(
        os.path.join(snap, "model.pth"), map_location="cpu", weights_only=False, mmap=True
    )
    if isinstance(sd, dict) and "model" in sd and not any(k.startswith("layers.") for k in sd):
        sd = sd["model"]
    return params, sd


def derive(params):
    dim = params["dim"]
    head_dim = params["head_dim"]
    n_heads = params["n_heads"] or dim // head_dim
    n_kv = params["n_kv_heads"] or n_heads
    mult = params.get("multiple_of", 256)
    ffn = int(params["ffn_dim_multiplier"] * dim)
    inter = mult * ((ffn + mult - 1) // mult)
    n_layers = params["n_layers"]
    moe_start = params.get("moe_start_from_layer", 0)
    moe_end = params.get("moe_end_from_layer", 0)
    n_exp = params.get("moe_n_experts", 1)
    topk = params.get("moe_router_topk", 1)
    special = {int(k): int(v) for k, v in (params.get("special_topk_layers") or {}).items()}
    # expert_used per layer: 0 = dense FFN, else MoE top-k
    eu = []
    for L in range(n_layers):
        is_moe = (n_exp > 1) and (L >= moe_start) and ((n_layers - L) > moe_end)
        eu.append(special.get(L, topk) if is_moe else 0)
    return dict(
        dim=dim, head_dim=head_dim, n_heads=n_heads, n_kv=n_kv, inter=inter,
        n_layers=n_layers, n_exp=n_exp, eu=eu,
    )


# Tensors kept near-lossless even in a quantized file: the embedding tables, the
# output head, and the router weights. Low-bit embeddings/head are where quality
# drops first, and the router selects experts (a discrete argmax over 16) so it is
# the most quant-sensitive matmul in the net. Everything else takes the bulk quant.
def is_high_precision(name):
    if name in ("output.weight", "text_embd.weight"):
        return True
    if name.startswith("audio_embd."):
        return True
    if name.startswith("blk.") and (".router_down." in name or ".router_mlp" in name):
        return True
    return False


def pick_qtype(name, ndim, ne0, outtype):
    """Per-tensor storage type. 1-D tensors (norms/biases/temp/eda-scale) are always
    f32. In a quantized file the bulk of the matrices take `outtype`, but the
    high-precision set (embeddings/head/router) is bumped one tier up. Returns
    'f32', 'f16', or a GGMLQuantizationType.

    Today the bulk quant is q8_0, so the bump is to f16 -- the only higher tier wired
    in pure Python, since gguf's K-quant quantizers raise NotImplementedError. When
    q4_k is added (via ggml_quantize_chunk over ctypes) the bulk becomes Q4_K and this
    same set falls back to Q8_0; only the q8_0 branch below grows a q4_k sibling."""
    if ndim == 1:
        return "f32"
    if outtype == "f32":
        return "f32"
    if outtype == "f16":
        return "f16"
    if outtype == "q8_0":
        if is_high_precision(name):
            return "f16"
        return GGMLQuantizationType.Q8_0 if ne0 % 32 == 0 else "f16"
    raise SystemExit(f"unknown outtype {outtype}")


def add_tensor(w, name, t, qtype):
    """Write tensor `t` with a concrete storage type from pick_qtype: 'f32', 'f16',
    or a GGMLQuantizationType. gguf.add_tensor does NOT quantize, so quantized types
    are pre-quantized to bytes here (per-row over the last/ne0 axis)."""
    t = t.contiguous()
    f = np.ascontiguousarray(t.float().numpy().astype(np.float32))
    if qtype == "f32":
        w.add_tensor(name, f)
    elif qtype == "f16":
        mx = float(np.abs(f).max())
        if mx >= 65504.0:
            raise SystemExit(f"F16 overflow in {name}: max|x|={mx}")
        w.add_tensor(name, f.astype(np.float16))
    elif f.ndim == 2:
        w.add_tensor(name, gquants.quantize(f, qtype), raw_dtype=qtype)
    elif f.ndim == 3:  # experts [E, out, in] -> quantize rows, keep 3-D byte shape
        E, out, inn = f.shape
        qd = gquants.quantize(f.reshape(E * out, inn), qtype).reshape(E, out, -1)
        w.add_tensor(name, qd, raw_dtype=qtype)
    else:
        w.add_tensor(name, f)  # scalar fallback (1-D already returned f32 above)
    return tuple(f.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="HF snapshot dir with model.pth + params.json")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--outtype", choices=["f16", "f32", "q8_0"], default="f16")
    args = ap.parse_args()

    params, sd = load_state(args.snapshot)
    h = derive(params)
    log("config:", {k: h[k] for k in ("dim", "head_dim", "n_heads", "n_kv", "inter", "n_layers", "n_exp")})
    log("expert_used_per_layer:", h["eu"])

    consumed = set()

    def g(key):
        consumed.add(key)
        return sd[key]

    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "ZONOS2")
    w.add_uint32(f"{ARCH}.block_count", h["n_layers"])
    w.add_uint32(f"{ARCH}.context_length", params["max_seqlen"])
    w.add_uint32(f"{ARCH}.embedding_length", h["dim"])
    w.add_uint32(f"{ARCH}.head_dim", h["head_dim"])
    w.add_uint32(f"{ARCH}.attention.head_count", h["n_heads"])
    w.add_uint32(f"{ARCH}.attention.head_count_kv", h["n_kv"])
    w.add_uint32(f"{ARCH}.feed_forward_length", h["inter"])
    w.add_float32(f"{ARCH}.rope.freq_base", float(params["rope_theta"]))
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", float(params["norm_eps"]))
    w.add_float32(f"{ARCH}.attention.qk_norm_epsilon", 1e-6)
    w.add_float32(f"{ARCH}.logit_softcap", float(params["loss_softcap"]))
    w.add_uint32(f"{ARCH}.expert_count", h["n_exp"])
    w.add_array(f"{ARCH}.expert_used_count_per_layer", h["eu"])
    w.add_uint32(f"{ARCH}.n_codebooks", params["n_codebooks"])
    w.add_uint32(f"{ARCH}.codebook_size", params["codebook_size"])
    w.add_uint32(f"{ARCH}.audio_vocab", params["codebook_size"] + 2)
    w.add_uint32(f"{ARCH}.eoa_id", params["eoa_id"])
    w.add_uint32(f"{ARCH}.audio_pad_id", params["audio_pad_id"])
    w.add_uint32(f"{ARCH}.text_vocab", params["text_vocab"])
    w.add_uint32(f"{ARCH}.speaker.embedding_dim", params["speaker_embedding_dim"])
    w.add_uint32(f"{ARCH}.speaker.lda_dim", params.get("speaker_lda_dim") or 0)
    w.add_file_type({"f32": 0, "f16": 1, "q8_0": 7}[args.outtype])

    added = {}
    type_counts = {}

    def add(name, t):
        qt = pick_qtype(name, t.dim(), int(t.shape[-1]), args.outtype)
        label = qt if isinstance(qt, str) else qt.name.lower()
        type_counts[label] = type_counts.get(label, 0) + 1
        added[name] = add_tensor(w, name, t, qt)

    nc = params["n_codebooks"]
    for i in range(nc):
        add(f"audio_embd.{i}.weight", g(f"multi_embedder.embedders.{i}.weight"))
    add("text_embd.weight", g(f"multi_embedder.embedders.{nc}.weight"))
    add("spk_lda.weight", g("speaker_lda_projection.weight"))
    add("spk_lda.bias", g("speaker_lda_projection.bias"))
    add("spk_proj.weight", g("speaker_projection.weight"))
    add("spk_proj.bias", g("speaker_projection.bias"))
    add("output_norm.weight", g("out_norm.weight"))
    add("output.weight", g("multi_output.weight"))

    for L in range(h["n_layers"]):
        p, b = f"layers.{L}.", f"blk.{L}."
        add(b + "attn_norm.weight", g(p + "attention_norm.weight"))
        add(b + "ffn_norm.weight", g(p + "ffn_norm.weight"))
        add(b + "attn_q.weight", g(p + "attention.wq.weight"))
        wkv = g(p + "attention.wkv.weight")  # [2, kv_dim, dim]: chunk0=K, chunk1=V
        add(b + "attn_k.weight", wkv[0])
        add(b + "attn_v.weight", wkv[1])
        add(b + "attn_output.weight", g(p + "attention.wo.weight"))
        add(b + "attn_temp", g(p + "attention.temp").reshape(-1).abs())  # [n_head], |temp| folded
        add(b + "attn_gate.weight", g(p + "attention.gater.weight"))

        if h["eu"][L] == 0:  # dense FFN
            win = g(p + "feed_forward.w_in.weight")  # [2, inter, dim]: chunk0=up, chunk1=gate
            add(b + "ffn_up.weight", win[0])
            add(b + "ffn_gate.weight", win[1])
            add(b + "ffn_down.weight", g(p + "feed_forward.w_out.weight"))
        else:  # MoE (sonic): w13 interleaved gate(even)/up(odd)
            w13 = g(p + "feed_forward.experts.w13")  # [E, 2*inter, dim]
            add(b + "ffn_gate_exps.weight", w13[:, 0::2, :])
            add(b + "ffn_up_exps.weight", w13[:, 1::2, :])
            add(b + "ffn_down_exps.weight", g(p + "feed_forward.experts.w2"))
            r = p + "feed_forward.router."
            add(b + "router_down.weight", g(r + "down_proj.weight"))
            add(b + "router_down.bias", g(r + "down_proj.bias"))
            add(b + "router_mlp0.weight", g(r + "router_mlp.0.weight"))
            add(b + "router_mlp0.bias", g(r + "router_mlp.0.bias"))
            add(b + "router_mlp2.weight", g(r + "router_mlp.2.weight"))
            add(b + "router_mlp2.bias", g(r + "router_mlp.2.bias"))
            add(b + "router_mlp4.weight", g(r + "router_mlp.4.weight"))
            add(b + "router_norm.weight", g(r + "rmsnorm_eda.weight"))
            add(b + "router_bias", g(r + "balancing_biases"))
            sk = r + "router_states_scale"
            if sk in sd:  # absent on the first MoE layer (no EDA)
                add(b + "router_eda_scale", g(sk))

    leftover = set(sd) - consumed
    if leftover:
        raise SystemExit(f"{len(leftover)} source keys not mapped, e.g.: {sorted(leftover)[:8]}")
    log(f"mapped all {len(consumed)} source keys -> {len(added)} gguf tensors")
    log("storage types:", dict(sorted(type_counts.items())))

    log(f"writing {args.out} ...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    log("done.")


if __name__ == "__main__":
    main()
