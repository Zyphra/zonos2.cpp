#!/usr/bin/env python3
"""Dump the ZONOS2 checkpoint key/shape/dtype inventory (ground truth for the converter).

Run with the project venv:
    /data/home/sofian/ZONOS2/.venv/bin/python models/inspect-checkpoint.py
"""
import json
import os
import sys

SNAP = "/data/home/sofian/.cache/huggingface/hub/models--Zyphra--ZONOS2/snapshots/0cc1f131a87b41141f556a5aef8a7b775c1c3ea1"
OUT = os.path.join(os.path.dirname(__file__), "..", "out", "checkpoint-keys.txt")


def main():
    import torch  # slow import on this box (~170s, Lustre venv)

    with open(os.path.join(SNAP, "params.json")) as f:
        params = json.load(f)
    print("params.json:", json.dumps(params, indent=0)[:400], "...")

    path = os.path.join(SNAP, "model.pth")
    print("loading (mmap)", path, "...", flush=True)
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(state, dict) and "model" in state and not any(
        k.startswith("layers.") for k in state
    ):
        state = state["model"]

    keys = sorted(state.keys())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for k in keys:
            t = state[k]
            f.write(f"{k}\t{tuple(t.shape)}\t{t.dtype}\n")
    print(f"\nwrote {len(keys)} keys -> {OUT}")

    # focused checks on the tricky tensors
    def show(k):
        if k in state:
            t = state[k]
            print(f"  {k:55s} {tuple(t.shape)} {t.dtype}")
        else:
            print(f"  {k:55s} <MISSING>")

    print("\n=== tricky tensors ===")
    for k in [
        "multi_embedder.embedders.0.weight",
        "multi_embedder.embedders.8.weight",
        "multi_embedder.embedders.9.weight",
        "multi_embedder.embedders.10.weight",
        "emb_norm.weight",
        "speaker_lda_projection.weight", "speaker_lda_projection.bias",
        "speaker_projection.weight", "speaker_projection.bias",
        "out_norm.weight", "multi_output.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wkv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.attention.temp",
        "layers.0.attention.gater.weight",
        "layers.0.attention_norm.weight", "layers.0.ffn_norm.weight",
        "layers.0.feed_forward.w_in.weight", "layers.0.feed_forward.w_out.weight",
        "layers.2.feed_forward.w_in.weight",   # last dense before MoE
        "layers.27.feed_forward.w_in.weight",  # last dense
        "layers.3.feed_forward.experts.w13",
        "layers.3.feed_forward.experts.w2",
        "layers.3.feed_forward.router.down_proj.weight",
        "layers.3.feed_forward.router.down_proj.bias",
        "layers.3.feed_forward.router.router_mlp.0.weight",
        "layers.3.feed_forward.router.router_mlp.0.bias",
        "layers.3.feed_forward.router.router_mlp.2.weight",
        "layers.3.feed_forward.router.router_mlp.4.weight",
        "layers.3.feed_forward.router.rmsnorm_eda.weight",
        "layers.3.feed_forward.router.balancing_biases",
        "layers.3.feed_forward.router.router_states_scale",  # expect MISSING (first MoE)
        "layers.4.feed_forward.router.router_states_scale",  # expect present (EDA)
        "layers.26.feed_forward.experts.w13",
        "layers.26.feed_forward.router.router_states_scale",
    ]:
        show(k)

    # detect which layers are MoE (have experts) vs dense (have w_in)
    moe = sorted({int(k.split(".")[1]) for k in state if ".feed_forward.experts." in k})
    dense = sorted({int(k.split(".")[1]) for k in state if ".feed_forward.w_in" in k})
    eda = sorted({int(k.split(".")[1]) for k in state if "router_states_scale" in k})
    print("\nMoE layers:", moe)
    print("dense layers:", dense)
    print("EDA (router_states_scale) layers:", eda)

    # any unexpected key prefixes?
    prefixes = sorted({k.split(".")[0] for k in keys})
    print("top-level prefixes:", prefixes)


if __name__ == "__main__":
    sys.exit(main())
