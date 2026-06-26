#!/usr/bin/env python3
"""Emit a tiny ZONOS2 GGUF with random weights — for CI smoke-testing only.

The real checkpoint is ~15 GB, far too large for CI. This builds a structurally
faithful but minuscule model (a few MB) so the release smoke test can actually run
a forward pass through zonos2-cli (prompt build -> backbone -> sampler), not just
check the usage string. The audio is gibberish; the point is exercising the load +
graph-build + decode path on every platform's release binary.

Dimensions that the prompt builder and output head index into are kept at their real
values (n_codebooks, audio_vocab, text_vocab, codebook ids); everything that only
affects compute cost (n_embd, n_ff, n_layer, n_head, n_expert, router/speaker dims)
is shrunk. Tensor names/shapes mirror models/convert-zonos2-to-gguf.py exactly.

Usage:  python scripts/make-dummy-model.py -o dummy.gguf
"""
from __future__ import annotations

import argparse

import numpy as np
from gguf import GGUFWriter

ARCH = "zonos2"

# --- shrunk compute dims (free to change) ---
DIM       = 32          # n_embd
HEAD_DIM  = 16
N_HEAD    = 2           # N_HEAD*HEAD_DIM == DIM
N_KV      = 1           # GQA: divides N_HEAD
INTER     = 128         # feed_forward_length
N_EXPERT  = 4
ROUTER    = 32          # router hidden dim
SPK_DIM   = 32          # speaker.embedding_dim
LDA_DIM   = 16          # speaker.lda_dim
# expert_used per layer: 0 = dense FFN, else MoE top-k. This mix exercises a dense
# layer, the first MoE layer (no EDA scale), a multi-expert MoE layer, and a trailing
# dense layer — the distinct code paths in the graph builder and loader.
EXPERT_USED = [0, 1, 2, 0]
N_LAYER   = len(EXPERT_USED)
N_CTX     = 512

# --- real dims the runtime indexes into (must match the reference) ---
N_CODEBOOKS = 9
CODEBOOK    = 1024
AUDIO_VOCAB = CODEBOOK + 2     # 1026: +eoa, +pad
EOA_ID      = CODEBOOK         # 1024
PAD_ID      = CODEBOOK + 1     # 1025
TEXT_VOCAB  = 519              # text_embd has TEXT_VOCAB+1 rows (the +1 is the pad row)
OUT_VOCAB   = N_CODEBOOKS * AUDIO_VOCAB   # 9234, the multi-output head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(0)

    def rnd(*shape):
        # small init so a 4-layer random net stays numerically tame (no NaN/Inf
        # through the RMS norms and the tanh logit softcap). 1-D tensors (norms,
        # biases, temp, eda-scale) keep f32; matrices store f16 to halve the
        # committed fixture, mirroring the real converter's storage scheme.
        x = (rng.standard_normal(shape) * 0.02)
        return x.astype(np.float32 if len(shape) == 1 else np.float16)

    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "ZONOS2-dummy")
    w.add_uint32(f"{ARCH}.block_count", N_LAYER)
    w.add_uint32(f"{ARCH}.context_length", N_CTX)
    w.add_uint32(f"{ARCH}.embedding_length", DIM)
    w.add_uint32(f"{ARCH}.head_dim", HEAD_DIM)
    w.add_uint32(f"{ARCH}.attention.head_count", N_HEAD)
    w.add_uint32(f"{ARCH}.attention.head_count_kv", N_KV)
    w.add_uint32(f"{ARCH}.feed_forward_length", INTER)
    w.add_float32(f"{ARCH}.rope.freq_base", 10000.0)
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_float32(f"{ARCH}.attention.qk_norm_epsilon", 1e-6)
    w.add_float32(f"{ARCH}.logit_softcap", 15.0)
    w.add_uint32(f"{ARCH}.expert_count", N_EXPERT)
    w.add_array(f"{ARCH}.expert_used_count_per_layer", EXPERT_USED)
    w.add_uint32(f"{ARCH}.n_codebooks", N_CODEBOOKS)
    w.add_uint32(f"{ARCH}.codebook_size", CODEBOOK)
    w.add_uint32(f"{ARCH}.audio_vocab", AUDIO_VOCAB)
    w.add_uint32(f"{ARCH}.eoa_id", EOA_ID)
    w.add_uint32(f"{ARCH}.audio_pad_id", PAD_ID)
    w.add_uint32(f"{ARCH}.text_vocab", TEXT_VOCAB)
    w.add_uint32(f"{ARCH}.speaker.embedding_dim", SPK_DIM)
    w.add_uint32(f"{ARCH}.speaker.lda_dim", LDA_DIM)
    w.add_file_type(1)  # f16 matrices, f32 1-D tensors

    # shapes below are numpy/torch order ([out, in], [rows, dim]); GGUFWriter stores
    # them reversed as ggml ne, matching convert-zonos2-to-gguf.py.
    for i in range(N_CODEBOOKS):
        w.add_tensor(f"audio_embd.{i}.weight", rnd(AUDIO_VOCAB, DIM))
    w.add_tensor("text_embd.weight", rnd(TEXT_VOCAB + 1, DIM))
    w.add_tensor("spk_lda.weight",  rnd(LDA_DIM, SPK_DIM))
    w.add_tensor("spk_lda.bias",    rnd(LDA_DIM))
    w.add_tensor("spk_proj.weight", rnd(DIM, LDA_DIM))
    w.add_tensor("spk_proj.bias",   rnd(DIM))
    w.add_tensor("output_norm.weight", rnd(DIM))
    w.add_tensor("output.weight",      rnd(OUT_VOCAB, DIM))

    seen_moe = False
    for L in range(N_LAYER):
        b = f"blk.{L}."
        w.add_tensor(b + "attn_norm.weight", rnd(DIM))
        w.add_tensor(b + "ffn_norm.weight",  rnd(DIM))
        w.add_tensor(b + "attn_q.weight",      rnd(N_HEAD * HEAD_DIM, DIM))
        w.add_tensor(b + "attn_k.weight",      rnd(N_KV   * HEAD_DIM, DIM))
        w.add_tensor(b + "attn_v.weight",      rnd(N_KV   * HEAD_DIM, DIM))
        w.add_tensor(b + "attn_output.weight", rnd(DIM, N_HEAD * HEAD_DIM))
        w.add_tensor(b + "attn_temp",          rnd(N_HEAD))
        w.add_tensor(b + "attn_gate.weight",   rnd(N_HEAD, DIM))

        if EXPERT_USED[L] == 0:  # dense FFN
            w.add_tensor(b + "ffn_up.weight",   rnd(INTER, DIM))
            w.add_tensor(b + "ffn_gate.weight", rnd(INTER, DIM))
            w.add_tensor(b + "ffn_down.weight", rnd(DIM, INTER))
        else:                    # MoE
            w.add_tensor(b + "ffn_gate_exps.weight", rnd(N_EXPERT, INTER, DIM))
            w.add_tensor(b + "ffn_up_exps.weight",   rnd(N_EXPERT, INTER, DIM))
            w.add_tensor(b + "ffn_down_exps.weight", rnd(N_EXPERT, DIM, INTER))
            w.add_tensor(b + "router_down.weight", rnd(ROUTER, DIM))
            w.add_tensor(b + "router_down.bias",   rnd(ROUTER))
            w.add_tensor(b + "router_mlp0.weight", rnd(ROUTER, ROUTER))
            w.add_tensor(b + "router_mlp0.bias",   rnd(ROUTER))
            w.add_tensor(b + "router_mlp2.weight", rnd(ROUTER, ROUTER))
            w.add_tensor(b + "router_mlp2.bias",   rnd(ROUTER))
            w.add_tensor(b + "router_mlp4.weight", rnd(N_EXPERT, ROUTER))
            w.add_tensor(b + "router_norm.weight", rnd(ROUTER))
            w.add_tensor(b + "router_bias",        rnd(N_EXPERT))
            if seen_moe:  # first MoE layer has no EDA scale (mirrors the real model)
                w.add_tensor(b + "router_eda_scale", rnd(ROUTER))
            seen_moe = True

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
