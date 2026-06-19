# zonos2.cpp

A standalone [ggml](https://github.com/ggml-org/ggml)/GGUF C++ port of the **ZONOS2**
text-to-speech model ([Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2), ~7.6B-param
MoE). The **entire pipeline** — speaker encoder, language-model backbone, and the audio
vocoder — runs as native C++ linking only `libggml` + `gguf`. **No Python, no PyTorch, no
CUDA-only kernels (flashinfer / sgl_kernel / cutlass / TVM) at inference time.**

```
  text ──▶ prompt builder ──▶ ZONOS2 backbone ──▶ 9 audio codes/frame ──▶ DAC ──▶ 44.1 kHz WAV
                                    ▲
  voice.mp3 ──▶ ECAPA speaker encoder ──▶ [2048] x-vector  (optional, for cloning)
```

One command turns text into a waveform:

```bash
zonos2-cli out/zonos2-q8_0.gguf --tts "Hello, world." out.wav \
    --dac out/dac.gguf --gpu --seed 1
```

## Highlights

- **Fully Python-free inference.** Three ggml graphs replace the whole reference stack:
  the ECAPA-TDNN speaker encoder (`wav → [2048]`), the 28-layer MoE backbone
  (`text + speaker → audio codes`), and the DAC-44 kHz decoder (`codes → waveform`).
- **CPU and CUDA**, same GGUF files. CUDA targets sm_90 (H100) by default.
- **Real-time on GPU:** ~300 fps decode, **RTF ≈ 0.28–0.32** (~3.5× faster than real-time)
  via a fused `flash_attn_ext` decode step, an F16 KV cache, and CUDA-graph replay.
- **Numerically validated against the PyTorch reference** at every stage — the backbone to
  cosine ≥ 0.9999 / matching argmax, the speaker encoder and DAC decoder **bit-exact**.
- **Quantization:** F16 (lossless from the bf16 checkpoint) and **Q8_0** (7.7 GB, with the
  quant-sensitive matrices kept at F16). Q4_K is the one remaining TODO.

## Repository layout

```
src/
  zonos2.{h,cpp}          GGUF loader, hparams, tensor map
  zonos2-graph.cpp        backbone graph (prefill/validate) + KV-cache decode + zonos2_generate
  zonos2-sampler.{h,cpp}  per-codebook sampler (temp/top-k/top-p/min-p, rep penalty, EOS)
  zonos2-prompt.cpp       text → input-id prompt (mirrors tts/prompt.py + scheduler)
  spk-encoder.cpp         ECAPA-TDNN speaker encoder → spk-encoder-cli
  dac.{h,cpp}             DAC-44kHz decoder library
  dac-cli.cpp             standalone codes → wav CLI
  main.cpp                zonos2-cli (summary / validate / generate / tts / build-prompt)
  npy.h                   tiny .npy reader/writer
models/                   GGUF converters + the PyTorch validation harness (see below)
ggml/                     vendored submodule (pinned 3af5f57)
out/                      generated GGUFs + golden/validation data (git-ignored)
```

## Build

### Prebuilt binaries

Tagged releases ship self-contained binaries (statically linked against `libggml`)
on the [Releases](../../releases) page, built by CI for: Linux x64 (CPU and Vulkan),
macOS arm64 (Metal), and Windows x64 (Vulkan). Each archive holds `zonos2-cli`,
`spk-encoder-cli`, and `dac-cli` — no shared-library install needed. The model GGUFs
are *not* bundled; convert or download them separately (see [Models](#models-one-time-conversion)).
Vulkan builds need a Vulkan-capable GPU driver at runtime. For CUDA (sm_90/H100),
build from source as below.

### From source

```bash
git clone --recurse-submodules <repo> zonos2.cpp && cd zonos2.cpp
# (already cloned? git submodule update --init)

# CPU
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# CUDA (H100 = sm_90; pass -DCMAKE_CUDA_ARCHITECTURES=<n> for other GPUs)
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

Each build produces three executables: **`zonos2-cli`**, **`spk-encoder-cli`**, **`dac-cli`**.
CUDA-graph replay (needed for the real-time decode) is enabled automatically for CUDA builds.

## Models (one-time conversion)

### Prebuilt GGUFs (Hugging Face)

Skip the conversion below by pulling the ready-made non-quant GGUFs from
[`Zyphra/ZONOS2-GGUF`](https://huggingface.co/Zyphra/ZONOS2-GGUF) — the F16 backbone plus
the DAC and speaker-encoder files (identical to what the converter emits):

```bash
hf download Zyphra/ZONOS2-GGUF zonos2-f16.gguf dac.gguf spk-encoder.gguf --local-dir out
```

From the F16 backbone you can make any quantization locally with `quantize-cli` — no
checkpoint or Python required (see [Quantization](#quantization)).

### Converting from the checkpoints

Inference is Python-free, but you convert the original checkpoints to GGUF once. Two
Python environments are involved, and they are **not** the same:

- **Conversion** (the `convert-*.py` below) needs only `torch` + `numpy` (plus
  `safetensors` + `torchaudio` for the speaker encoder). `gguf` is **vendored** in
  `models/_pydeps/`, so don't `pip install gguf`. A throwaway venv is enough:
  ```bash
  python3 -m venv .venv && . .venv/bin/activate
  pip install torch numpy safetensors torchaudio
  ```
- **Validation** (`models/dump-*-golden.py`, see below) additionally imports the real
  `zonos2` package, so it needs the full [ZONOS2 reference repo](https://huggingface.co/Zyphra/ZONOS2)
  installed in its own venv. On the dev node that's `/data/home/sofian/ZONOS2/.venv`.

### Obtaining the source checkpoints

The converters take a local path to each checkpoint. Fetch them once (sizes are the
download, not the GGUF output):

```bash
# 1) ZONOS2 backbone (model.pth + params.json). Accept the license / `huggingface-cli login`
#    first if the repo is gated. Prints the local snapshot dir the converter wants.
pip install -U "huggingface_hub[cli]"
ZB=$(huggingface-cli download Zyphra/ZONOS2)

# 2) Speaker encoder — HF repo is misnamed "Qwen3-Voice-Embedding" but ships the ECAPA-TDNN.
ZS=$(huggingface-cli download marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B)

# 3) DAC 44 kHz vocoder weights → ~/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth
pip install descript-audio-codec
python3 -m dac download --model_type 44khz
```

> The hardcoded `~/.cache/huggingface/hub/models--.../snapshots/<hash>` paths in older
> command examples are just what `huggingface-cli download` populates — use the `$ZB` /
> `$ZS` it prints instead of pinning a snapshot hash.

### Converting to GGUF

```bash
PY=/data/home/sofian/ZONOS2/.venv/bin/python   # or your conversion venv's python

# 1) Backbone — F16 (15 GB, lossless) or Q8_0 (7.7 GB). $ZB from "Obtaining" above.
$PY models/convert-zonos2-to-gguf.py $ZB --outtype f16  -o out/zonos2-f16.gguf
$PY models/convert-zonos2-to-gguf.py $ZB --outtype q8_0 -o out/zonos2-q8_0.gguf

# 2) Speaker encoder — ECAPA-TDNN, 24 MB. $ZS from "Obtaining" above.
$PY models/convert-spk-encoder-to-gguf.py $ZS -o out/spk-encoder.gguf

# 3) DAC 44 kHz vocoder decoder — 254 MB (all f32).
$PY models/convert-dac-to-gguf.py ~/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth -o out/dac.gguf
```

| GGUF | size | contents |
|---|---|---|
| `zonos2-f16.gguf`  | 15 GB | backbone, F16 (lossless vs bf16) |
| `zonos2-q8_0.gguf` | 7.7 GB | backbone, Q8_0 bulk + F16 embeddings/head/router |
| `spk-encoder.gguf` | 24 MB | ECAPA-TDNN encoder (F16 convs) |
| `dac.gguf`         | 254 MB | DAC-44kHz decoder (F32) |

## Usage

### Text → speech (one command)

```bash
zonos2-cli out/zonos2-q8_0.gguf --tts "Hello, world." out.wav \
    --dac out/dac.gguf --gpu --seed 1
```

With `--dac`, a `.wav` output is decoded directly; an `.npy` output writes the raw codes
(plus a sibling `.eos.npy`) and, if `--dac` is given, a sibling `.wav`.

> ⚠️ **Use sampling, not greedy.** A `--seed` (or omitting `--greedy`) lets the model emit
> EOS and produce a finite clip. With `--greedy` the backbone never emits EOS, runs to
> `--max`, and the audio comes out near-silent. This is a property of the original model,
> not the port.

### Voice cloning (two commands)

```bash
# 1) Extract a [2048] speaker embedding from any audio file (--clone shells ffmpeg in-process).
spk-encoder-cli out/spk-encoder.gguf --clone voice.mp3 emb.npy

# 2) Synthesize in that voice.
zonos2-cli out/zonos2-q8_0.gguf --tts "Cloned voice demo." out.wav \
    --dac out/dac.gguf --speaker emb.npy --gpu --seed 1
```

The repo's three bundled reference voices live in `ZONOS2/default_voices/*.mp3`. The speaker
encoder also accepts `--wav <24kHz-mono.npy>`, `--raw <f32le>`, or a precomputed
`--mel <[T,128].npy>`.

### Standalone components

```bash
# codes → wav directly
dac-cli out/dac.gguf out/codes.npy out.wav --gpu          # reads out/codes.npy.eos.npy if present

# build the prompt id matrix without generating
zonos2-cli out/zonos2-f16.gguf --build-prompt "Hello." out/ids.npy

# generate from a precomputed prompt (.npy of input ids)
zonos2-cli out/zonos2-q8_0.gguf --generate out/ids.npy out/codes.npy --gpu --seed 1
```

### Useful flags (`zonos2-cli`)

| flag | meaning |
|---|---|
| `--cpu` / `--gpu` | backend (default CPU) |
| `--seed N` | sampler seed (enables sampling) |
| `--greedy` | greedy decode (deterministic; see EOS caveat) |
| `--max N` | max frames (default 400) |
| `--speaker emb.npy [--speaker-pos P]` | inject a voice embedding (default position 0) |
| `--dac dac.gguf` | decode codes to a WAV in the same run |
| `--recompute` | O(n²) reference decode instead of the KV cache (for checking) |

### Validation against the reference

The `models/` harness reproduces the PyTorch forward pass and diffs it against the C++ graph:

```bash
PY=/data/home/sofian/ZONOS2/.venv/bin/python
$PY models/dump-golden.py --prompt "Hello, world." -o out/golden          # PyTorch intermediates
zonos2-cli out/zonos2-f16.gguf --validate out/golden/input_ids.npy out/cpu --cpu
$PY models/compare.py out/golden out/cpu --cos 0.99                        # layer-by-layer diff
```

`dump-dac-golden.py` / `dump-spk-golden.py` do the same for the vocoder and encoder.

## Model architecture (what the port mirrors)

| | |
|---|---|
| Backbone | 28 layers; dense FFN on 0/1/2/27, **MoE on 3–26** |
| Dims | n_embd 2048, head_dim 128, 16 Q/O heads, **4 KV heads (GQA 4:1)** |
| MoE | 16 experts, top-1 (top-2 only on layer 26), `moe_impl="sonic"`, router_dim 128 |
| Attention | per-head QK-RMSNorm (eps 1e-6) + per-head temp, **interleaved RoPE**, per-head sigmoid output gate |
| Router | EDA router with depth-threaded `router_states`, exact erf-GELU, prob-weighted no-renorm top-k |
| Codebooks | 9 × 1026 (eoa 1024, pad 1025); output head 9×1026 with 15·tanh(x/15) softcap |
| Speaker | ECAPA-TDNN (~6M) → [2048] → LDA 1024 → proj 2048, **overwrites** embedding row 0 |
| Vocoder | DAC-44kHz 8kbps decode path: `quantizer.from_codes` + decoder, hop 512 |

See `~/.claude/plans/implement-this-model-in-mellow-nova.md` for the full spec, the
risk list, and the phase-by-phase validation that drove the port.

## Performance

Measured on one H100 with `zonos2-q8_0.gguf` (decode is the dominant cost):

- **~300 fps** greedy / ~270 fps sampling, **RTF ≈ 0.28–0.32**, flat with sequence length.
- The decode step is one fused `flash_attn_ext` over an **F16 KV cache**
  `[head_dim, max_seq, n_head_kv]`; the graph is built and allocated once and **replayed via
  CUDA graphs** (Q8_0 experts keep `mul_mat_id` capturable). Prefill/validate use the manual
  F32 attention path.

## Numerical validation

| stage | result vs PyTorch reference |
|---|---|
| Backbone logits (F16, CPU & CUDA) | cosine **0.999989**, last-position argmax **9/9** |
| Backbone + speaker inject | spk_proj 0.999996, logits 0.999985, argmax 100% |
| Q8_0 backbone vs golden | logits cos 0.999585, full-seq argmax **98.65%**, last-pos 100% |
| Speaker encoder (`wav → [2048]`) | **bit-exact** — cos 1.0000001, max\|Δ\| 2.98e-7 |
| DAC decoder (`codes → wav`), CPU & GPU | **bit-exact** — cos **1.0000000**, max\|Δ\| < int16 step |

## Quantization

- **F16** — lossless for these in-range bf16 weights; one file serves every backend. The
  published master on HF and the input to `quantize-cli`.
- **Q8_0** — bulk 2-D/3-D matrices at Q8_0, 1-D at F32, and the quant-sensitive tensors
  (embedding tables, output head, all router weights) bumped to F16. 7.7 GB, +41 MB over
  pure Q8_0, strictly better against golden; CUDA graphs still replay.
- **Q4_K and other K-quants** — produced by `quantize-cli` (below) via `ggml_quantize_chunk`.
  The pure-Python converter still can't emit K-quants (`gguf.quants` raises
  `NotImplementedError`), so quantize from the F16 GGUF instead.

### Making quants from the F16 GGUF

`quantize-cli` requantizes the F16 backbone to any ggml quant type — no checkpoint or Python
needed, so it runs straight off the HF download:

```bash
quantize-cli out/zonos2-f16.gguf out/zonos2-q4_k.gguf q4_k
# types: q8_0 q4_0 q4_1 q5_0 q5_1 q2_k q3_k q4_k q5_k q6_k iq4_nl iq4_xs
```

It mirrors the converter's per-tensor policy: 1-D tensors stay F32, the quant-sensitive set
(output head, token/audio embeddings, MoE routers) is kept one tier above the bulk quant, and
every other matrix takes the requested type — falling back to F16 if its row length isn't
block-aligned. The whole F16 file is loaded into RAM (~15 GB) alongside the output, so size
the machine accordingly. Quantizing from F16 (vs the bf16 checkpoint) is numerically
equivalent — f16 is lossless for these weights, so the result matches the converter's Q8_0 to
within quantizer rounding (≈1 element in 4M off by one LSB).

## Notes

- Activation/residual streams are F32; weights are F16/Q8_0/F32 per the tables above.
- Layout convention everywhere: **ggml `ne` = reversed `numpy.shape`**. Backbone and speaker
  activations are channel-major `[C,T]`; DAC activations are time-major `[T,C]` (what ggml's
  conv ops want).
- `ggml`, the build dirs, and the multi-GB GGUFs are git-ignored.
- These tools were developed on a node where `/data` is the working filesystem (keep the
  project, builds, and GGUFs there) and the GPUs are shared — pick a free one with
  `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` accordingly.

## Credits

- [ggml](https://github.com/ggml-org/ggml) — the tensor library / GGUF format.
- [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2) — the original model and PyTorch
  reference implementation that this port mirrors and is validated against.
- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) — the DAC vocoder.
