#!/usr/bin/env python3
"""Dump ground-truth intermediates from the REAL ZONOS2 model (no speaker) for a
fixed prompt, by monkeypatching submodule .forward during the (eager) prefill.

Captures the exact token sequence the model embeds (so the C++ side feeds the same
input), every module-boundary activation, and the final logits. Offline TTSLLM uses
no speaker conditioning, so pos-0 is not overwritten (the speaker path is unit-tested
separately).

    CUDA_VISIBLE_DEVICES=2 /data/home/sofian/ZONOS2/.venv/bin/python \
        models/dump-golden.py --prompt "Hello, world." -o out/golden
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

SNAP = "/data/home/sofian/.cache/huggingface/hub/models--Zyphra--ZONOS2/snapshots/0cc1f131a87b41141f556a5aef8a7b775c1c3ea1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Hello, world.")
    ap.add_argument("--model", default=SNAP)
    ap.add_argument("-o", "--out", default=os.path.join(os.path.dirname(__file__), "..", "out", "golden"))
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--speaker", default=None,
                    help='speaker embedding npy ([spk_dim] f32), or "auto" for a deterministic random one')
    ap.add_argument("--speaker-pos", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from zonos2.tts import TTSLLM
    from zonos2.message.tts import TTSSamplingParams

    print("building TTSLLM (loads model, warms up) ...", flush=True)
    tts = TTSLLM(model_path=args.model, decode_audio=False)
    model = tts.engine.model
    layers = model.layers.op_list
    n_layer = len(layers)
    print(f"model ready: {n_layer} layers", flush=True)

    cap = {}
    state = {"first": True}

    def npy(t):
        return t.detach().float().cpu().numpy()

    def save(name, t):
        if torch.is_tensor(t):
            cap[name] = npy(t)

    # IMPORTANT: snapshot inputs BEFORE calling the module — the fused MoE kernel
    # writes its output in place over its input buffer, so reading a[0] afterwards
    # would return the output, not the input.
    def patch(obj, cb, fn_name="forward"):
        orig = getattr(obj, fn_name)

        def w(*a, **k):
            pre = npy(a[0]) if (state["first"] and a and torch.is_tensor(a[0])) else None
            out = orig(*a, **k)
            if state["first"]:
                cb(pre, out)
            return out
        setattr(obj, fn_name, w)

    def out0(out):
        return out[0] if isinstance(out, tuple) else out

    # --- optional speaker: project a (fixed) speaker embedding via the REAL modules; the
    # projected vector overwrites the embedding row at speaker_pos (mirrors zonos2.py:850-866). ---
    spk_proj_vec = None
    if args.speaker:
        lda  = getattr(model, "speaker_lda_projection", None)
        proj = getattr(model, "speaker_projection", None)
        assert proj is not None, "model has no speaker_projection"
        wref = (lda.weight if lda is not None else proj.weight)  # model isn't an nn.Module
        dev, dt = wref.device, wref.dtype
        spk_dim = (lda.weight.shape[1] if lda is not None else proj.weight.shape[1])
        if args.speaker == "auto":
            spk_np = np.random.default_rng(0).standard_normal(spk_dim).astype(np.float32)
            np.save(os.path.join(args.out, "speaker.npy"), spk_np)
            print(f"speaker: generated deterministic {spk_dim}-d vector -> speaker.npy", flush=True)
        else:
            spk_np = np.load(args.speaker).astype(np.float32).reshape(-1)
            assert spk_np.shape[0] == spk_dim, f"speaker dim {spk_np.shape[0]} != {spk_dim}"
        with torch.no_grad():
            s = torch.from_numpy(spk_np).to(device=dev, dtype=dt).unsqueeze(0)  # [1, spk_dim]
            sl = lda.forward(s) if lda is not None else s
            spk_proj_vec = proj.forward(sl)[0]                                  # [hidden]
        cap["spk_proj"] = npy(spk_proj_vec)
        print(f"speaker: spk_proj computed, pos={args.speaker_pos}", flush=True)

    # multi_embedder: capture input_ids + emb_sum, then (if speaker) overwrite row spk_pos
    # and capture emb_after_spk. The overwritten embedding flows through the real network.
    orig_me = model.multi_embedder.forward
    def me_w(*a, **k):
        out = orig_me(*a, **k)
        if state["first"]:
            cap["input_ids"] = npy(a[0])
            save("emb_sum", out)
            if spk_proj_vec is not None:
                out = out.clone()
                out[args.speaker_pos] = spk_proj_vec.to(out.dtype)
                save("emb_after_spk", out)
        return out
    model.multi_embedder.forward = me_w
    patch(model.emb_norm,       lambda pre, out: save("emb_norm", out0(out)))

    def mk_attn(i):
        def cb(pre, out):
            cap[f"attn_in_{i}"] = pre
            save(f"attn_out_{i}", out)
        return cb

    def mk_ffn(i):
        def cb(pre, out):
            cap[f"ffn_in_{i}"] = pre
            save(f"ffn_out_{i}", out0(out))
            if isinstance(out, tuple):
                save(f"router_states_{i}", out[1])
        return cb

    def mk_layer(i):
        def cb(pre, out):
            save(f"layer_x_{i}", out[0]); save(f"layer_res_{i}", out[1])
        return cb

    for i, ly in enumerate(layers):
        patch(ly.attention,    mk_attn(i))
        patch(ly.feed_forward, mk_ffn(i))
        patch(ly,              mk_layer(i))

    patch(model.out_norm,     lambda pre, out: save("out_norm", out0(out)))
    patch(model.multi_output, lambda pre, out: save("mout", out))

    def logits_cb(pre, out):
        if pre is not None:
            cap["logits_in"] = pre
        save("logits", out)
        state["first"] = False  # stop capturing after the first full forward
    patch(model, logits_cb, "compute_logits")

    sp = TTSSamplingParams(seed=0, max_tokens=args.max_tokens)
    print("running prefill+decode ...", flush=True)
    res = tts.generate([args.prompt], sp)

    # persist
    for name, arr in cap.items():
        np.save(os.path.join(args.out, name + ".npy"), arr)
    meta = {
        "prompt": args.prompt,
        "n_tokens": int(cap["input_ids"].shape[0]),
        "frame_width": int(cap["input_ids"].shape[1]),
        "n_layer": n_layer,
        "expert_used": [int((ly.feed_forward.__class__.__name__ == "MoEFeedForward")) for ly in layers],
        "logits_shape": list(cap["logits"].shape),
        "first_frames": res[0]["audio_tokens"][:8],
        "eos_frame": res[0]["eos_frame"],
        "speaker": args.speaker,
        "speaker_pos": args.speaker_pos,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {len(cap)} tensors -> {args.out}")
    print("meta:", json.dumps(meta, indent=0)[:500])


if __name__ == "__main__":
    main()
