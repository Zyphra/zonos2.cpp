#!/usr/bin/env python3
"""Convert the Descript Audio Codec (DAC 44kHz, 8kbps) DECODER to GGUF.

ZONOS2 emits 9 audio-codebook codes per frame; the only remaining Python piece of the
standalone port is turning those codes into a waveform via DAC. We port just the decode
path: quantizer.from_codes (codes -> latent z) + decoder (z -> 44.1kHz audio).

What we store (decode-only):
  * Quantizer, FOLDED. from_codes does, per codebook i:
        z_q += out_proj_i( codebook_i[code] )            (out_proj is a 1x1 conv = linear 8->1024)
    Since out_proj is linear we fold it into a per-codebook lookup table:
        table_i[c] = codebook_i[c] @ W_out_i^T           -> [codebook_size=1024, latent=1024]
        z_q[:,t]   = sum_i table_i[code_i,t] + (sum_i out_proj_i.bias)
    => 9 get_rows + adds + one total bias. (in_proj is encode-only; dropped.)
  * Decoder: WNConv1d in (1024->1536,k7), 4 DecoderBlocks (snake -> WNConvTranspose1d
    upsample -> 3 ResidualUnits dil 1/3/9), final snake -> WNConv1d (96->1,k7) -> tanh.

weight_norm is FOLDED at convert time via torch._weight_norm(v, g, dim=0) (dim=0 works for
both Conv1d weight [OC,IC,K] and ConvTranspose1d weight [IC,OC,K] -- g is [.,1,1]).

Layout rule: ggml ne = reversed(numpy.shape). Activations in C++ are time-major [T, C]
(ne0=time), which is what ggml_conv_1d / ggml_conv_transpose_1d want, so:
  * Conv1d weight  (eff [OC,IC,K]) stored as-is -> ggml ne=[K,IC,OC]
  * ConvT weight   (eff [IC,OC,K]) stored as-is -> ggml ne=[K,OC,IC]
  * conv/convT bias [OC]    -> numpy [OC,1] -> ggml ne=[1,OC]  (broadcast over time)
  * snake alpha [1,C,1]     -> numpy [C,1]  -> ggml ne=[1,C]
  * quant table             -> numpy [codes,latent] -> ggml ne=[latent,codes] (get_rows on ne1)
  * quant total bias        -> numpy [latent] (1-D)

    /data/home/sofian/ZONOS2/.venv/bin/python models/convert-dac-to-gguf.py \
        ~/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth -o out/dac.gguf
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_pydeps"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gguf import GGUFWriter  # noqa: E402

ARCH = "dac"


def log(*a):
    print(*a, flush=True)


def eff_weight(sd, prefix):
    """Materialize a weight_norm'd weight: torch._weight_norm(v, g, dim=0)."""
    g = sd[prefix + ".weight_g"]
    v = sd[prefix + ".weight_v"]
    return torch._weight_norm(v, g, 0).detach().cpu().float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="weights_44khz_8kbps_0.0.1.pth")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    kw = ck["metadata"]["kwargs"]
    log("kwargs:", kw)

    n_cb = kw["n_codebooks"]              # 9
    cb_size = kw["codebook_size"]         # 1024
    cb_dim = kw["codebook_dim"]           # 8
    dec_rates = kw["decoder_rates"]       # [8,8,4,2]
    dec_dim = kw["decoder_dim"]           # 1536
    enc_rates = kw["encoder_rates"]       # [2,4,8,8]
    latent = kw["encoder_dim"] * (2 ** len(enc_rates))  # 64*16 = 1024
    sr = kw["sample_rate"]                # 44100
    hop = int(np.prod(dec_rates))         # 512

    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "DAC-44kHz-8kbps-decoder")
    w.add_uint32(f"{ARCH}.n_codebooks", n_cb)
    w.add_uint32(f"{ARCH}.codebook_size", cb_size)
    w.add_uint32(f"{ARCH}.codebook_dim", cb_dim)
    w.add_uint32(f"{ARCH}.latent_dim", latent)
    w.add_uint32(f"{ARCH}.decoder_dim", dec_dim)
    w.add_array(f"{ARCH}.decoder_rates", dec_rates)
    w.add_uint32(f"{ARCH}.sample_rate", sr)
    w.add_uint32(f"{ARCH}.hop_length", hop)
    w.add_uint32(f"{ARCH}.audio_pad_id", 1025)

    # All tensors stored f32: DAC weights are trained f32, and the C++ conv helper feeds the
    # kernel as mul_mat's f32 src1 (im2col f32 is src0) to keep full precision + a [T,C] layout.
    n = 0

    def put(name, arr, dtype=np.float32):
        nonlocal n
        a = np.ascontiguousarray(arr.astype(dtype))
        w.add_tensor(name, a)
        n += 1

    # ---- quantizer (folded) ----
    total_bias = np.zeros((latent,), dtype=np.float64)
    for i in range(n_cb):
        q = f"quantizer.quantizers.{i}"
        cb = sd[f"{q}.codebook.weight"].detach().cpu().float().numpy()      # [cb_size, cb_dim]
        wo = eff_weight(sd, f"{q}.out_proj").squeeze(-1)                    # [latent, cb_dim] (k=1)
        bo = sd[f"{q}.out_proj.bias"].detach().cpu().float().numpy()        # [latent]
        table = cb @ wo.T                                                   # [cb_size, latent]
        put(f"quant.{i}.table", table)                                     # ggml ne=[latent, cb_size]
        total_bias += bo
    put("quant.bias", total_bias)                                          # 1-D [latent], f32

    # ---- decoder ----
    def conv(name, prefix):
        put(name + ".weight", eff_weight(sd, prefix))                       # [OC,IC,K] -> ne=[K,IC,OC]
        put(name + ".bias", sd[prefix + ".bias"].detach().cpu().float().numpy()[:, None])  # [OC,1]->ne=[1,OC]

    def convt(name, prefix):
        put(name + ".weight", eff_weight(sd, prefix))                       # [IC,OC,K] -> ne=[K,OC,IC]
        put(name + ".bias", sd[prefix + ".bias"].detach().cpu().float().numpy()[:, None])

    def snake(name, prefix):
        a = sd[prefix + ".alpha"].detach().cpu().float().numpy().reshape(-1, 1)  # [C,1]->ne=[1,C]
        put(name + ".alpha", a)
        put(name + ".inv_alpha", 1.0 / (a + 1e-9))

    # decoder.model.0: WNConv1d(latent->dec_dim, k7)
    conv("dec.conv_in", "decoder.model.0")

    # decoder.model.{1..4}: DecoderBlock(in -> in/2, stride=dec_rates[b])
    for b, stride in enumerate(dec_rates):
        m = f"decoder.model.{b + 1}"
        snake(f"dec.b{b}.snake", f"{m}.block.0")
        convt(f"dec.b{b}.convt", f"{m}.block.1")
        # 3 ResidualUnits at .block.{2,3,4}, dilations 1/3/9
        for r, dil in enumerate([1, 3, 9]):
            ru = f"{m}.block.{r + 2}"
            snake(f"dec.b{b}.res{r}.snake1", f"{ru}.block.0")
            conv(f"dec.b{b}.res{r}.conv1", f"{ru}.block.1")   # k7, dilation dil
            snake(f"dec.b{b}.res{r}.snake2", f"{ru}.block.2")
            conv(f"dec.b{b}.res{r}.conv2", f"{ru}.block.3")   # k1

    # decoder.model.5: snake ; decoder.model.6: WNConv1d(96->1,k7) ; tanh
    snake("dec.snake_out", "decoder.model.5")
    conv("dec.conv_out", "decoder.model.6")

    log(f"latent={latent} dec_dim={dec_dim} rates={dec_rates} hop={hop} sr={sr}")
    log(f"wrote {n} tensors -> {args.out}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    log("done.")


if __name__ == "__main__":
    main()
