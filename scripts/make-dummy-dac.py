#!/usr/bin/env python3
"""Emit a tiny DAC GGUF with random weights — for CI smoke-testing only.

The real DAC decoder is ~254 MB; this builds a structurally faithful but minuscule
decoder (a few hundred KB) so the README/server tests can run the full codes -> waveform
path on CI without downloading the real vocoder. The audio is gibberish; the point is
exercising dac_load + the decode graph (quantizer from_codes -> conv stack -> tanh) on the
CPU backend.

Tensor names/shapes mirror models/convert-dac-to-gguf.py exactly (which is what src/dac.cpp
loads). Indexing dims the runtime needs are kept real (n_codebooks=9, codebook_size=1024 so
backbone codes index valid table rows); everything that only affects compute cost (latent,
decoder_dim, decoder_rates) is shrunk.

Usage:  python scripts/make-dummy-dac.py -o tests/fixtures/dummy-dac.gguf
"""
from __future__ import annotations

import argparse

import numpy as np
from gguf import GGUFWriter

ARCH = "dac"

# --- real dims the runtime indexes into (must match the backbone's codes) ---
N_CODEBOOKS = 9
CODEBOOK    = 1024            # codes are clamped to <= 1023, so tables need 1024 rows

# --- shrunk compute dims (free to change) ---
LATENT    = 8                # latent_dim (folded quantizer output / decoder input channels)
DEC_DIM   = 16              # decoder_dim; must be divisible by 2**len(DEC_RATES)
DEC_RATES = [2, 2]          # upsampling factors; hop_length = product(DEC_RATES)
SR        = 44100           # keep real so WAV headers are sane
HOP       = int(np.prod(DEC_RATES))
PAD_ID    = 1025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    assert DEC_DIM % (2 ** len(DEC_RATES)) == 0, "DEC_DIM must divide by 2**n_blocks"
    rng = np.random.default_rng(0)

    def rnd(*shape):
        # small init keeps a random net numerically tame (no NaN/Inf through snake's
        # 1/(alpha+eps)*sin^2 or the final tanh). All DAC tensors are f32 (matches the
        # real converter: weights are trained f32 and fed as mul_mat's f32 weights).
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    def alpha(c):
        # snake alpha well away from 0 so inv_alpha = 1/(alpha+1e-9) stays small.
        return (0.5 + 0.02 * rng.standard_normal((c, 1))).astype(np.float32)

    w = GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "DAC-dummy")
    w.add_uint32(f"{ARCH}.n_codebooks", N_CODEBOOKS)
    w.add_uint32(f"{ARCH}.codebook_size", CODEBOOK)
    w.add_uint32(f"{ARCH}.latent_dim", LATENT)
    w.add_uint32(f"{ARCH}.decoder_dim", DEC_DIM)
    w.add_array(f"{ARCH}.decoder_rates", DEC_RATES)
    w.add_uint32(f"{ARCH}.sample_rate", SR)
    w.add_uint32(f"{ARCH}.hop_length", HOP)
    w.add_uint32(f"{ARCH}.audio_pad_id", PAD_ID)

    n = 0

    def put(name, arr):
        nonlocal n
        w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)))
        n += 1

    # ---- quantizer (folded): 9 lookup tables [codes, latent] + one shared bias ----
    for i in range(N_CODEBOOKS):
        put(f"quant.{i}.table", rnd(CODEBOOK, LATENT))      # numpy [codes,latent] -> ne [latent,codes]
    put("quant.bias", rnd(LATENT))                          # 1-D [latent]

    # ---- decoder ----
    def conv(name, oc, ic, k):                              # Conv1d weight numpy [OC,IC,K], bias [OC,1]
        put(name + ".weight", rnd(oc, ic, k))
        put(name + ".bias", rnd(oc, 1))

    def convt(name, ic, oc, stride):                        # ConvTranspose1d weight numpy [IC,OC,K=2*stride]
        put(name + ".weight", rnd(ic, oc, 2 * stride))      # K must be 2*stride (src/dac.cpp asserts)
        put(name + ".bias", rnd(oc, 1))

    def snake(name, c):
        a = alpha(c)
        put(name + ".alpha", a)
        put(name + ".inv_alpha", 1.0 / (a + 1e-9))

    # decoder.model.0: conv_in (latent -> dec_dim, k7)
    conv("dec.conv_in", DEC_DIM, LATENT, 7)

    # decoder blocks: DecoderBlock(in -> in/2, stride=DEC_RATES[b])
    for b, stride in enumerate(DEC_RATES):
        ch_in = DEC_DIM >> b
        ch_out = DEC_DIM >> (b + 1)
        snake(f"dec.b{b}.snake", ch_in)                     # snake on block input
        convt(f"dec.b{b}.convt", ch_in, ch_out, stride)     # upsample, halve channels
        for r in range(3):                                  # 3 ResidualUnits, dilations 1/3/9
            snake(f"dec.b{b}.res{r}.snake1", ch_out)
            conv(f"dec.b{b}.res{r}.conv1", ch_out, ch_out, 7)
            snake(f"dec.b{b}.res{r}.snake2", ch_out)
            conv(f"dec.b{b}.res{r}.conv2", ch_out, ch_out, 1)

    final_ch = DEC_DIM >> len(DEC_RATES)
    snake("dec.snake_out", final_ch)
    conv("dec.conv_out", 1, final_ch, 7)                    # -> mono, then tanh

    print(f"latent={LATENT} dec_dim={DEC_DIM} rates={DEC_RATES} hop={HOP} -> {n} tensors -> {args.out}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()


if __name__ == "__main__":
    main()
