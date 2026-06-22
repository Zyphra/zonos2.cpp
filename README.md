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
- **HTTP server** (`zonos2-server`) mirroring the reference FastAPI: low-latency streaming
  PCM, OpenAI `/v1/audio/speech`, in-process reference-audio voice cloning, and a browser UI.
- **Numerically validated against the PyTorch reference** at every stage — the backbone to
  cosine ≥ 0.9999 / matching argmax, the speaker encoder and DAC decoder **bit-exact**.
- **Quantization:** F16 (lossless from the bf16 checkpoint) plus an **F16-spine expert ladder** —
  `quantize-cli --experts-only --spine-f16` keeps the whole spine (attention, dense FFN, router,
  embeddings, head) at F16 and K-quants only the MoE experts. The full-precision spine keeps the
  router on-distribution, so audio quality (WER/SpkSim/UTMOS) stays within eval noise of F16 all
  the way down to **Q4_K (4.9 GB)**. Quant quality is scored with the `zonos2-perplexity`
  KL-divergence tool.

## Repository layout

```
src/
  zonos2.{h,cpp}          GGUF loader, hparams, tensor map
  zonos2-graph.cpp        backbone graph (prefill/validate) + KV-cache decode + zonos2_generate
  zonos2-sampler.{h,cpp}  per-codebook sampler (temp/top-k/top-p/min-p, rep penalty, EOS)
  zonos2-prompt.cpp       text → input-id prompt (mirrors tts/prompt.py + scheduler)
  spk-encoder.{h,cpp}     ECAPA-TDNN speaker encoder library (shared by CLI + server)
  spk-encoder-cli.cpp     spk-encoder-cli (wav/mel/clone → [2048] embedding)
  dac.{h,cpp}             DAC-44kHz decoder library (full + windowed/streaming decode)
  dac-cli.cpp             standalone codes → wav CLI (+ --seam self-check)
  main.cpp                zonos2-cli (summary / validate / generate / tts / build-prompt)
  server.cpp              zonos2-server (HTTP TTS server; mirrors ../ZONOS2's FastAPI)
  quantize.cpp            gguf→gguf requantizer → quantize-cli (bulk or --experts-only)
  perplexity.cpp          KL-divergence / perplexity eval → zonos2-perplexity
  npy.h                   tiny .npy reader/writer
vendor/                   header-only deps: cpp-httplib, nlohmann/json (server only)
web/tts_ui.html           browser UI served by zonos2-server at /
models/                   GGUF converters + the PyTorch validation harness (see below)
ggml/                     vendored submodule (pinned 3af5f57)
out/                      generated GGUFs + golden/validation data (git-ignored)
```

## Build

### Prebuilt binaries

Tagged releases ship self-contained binaries (statically linked against `libggml`)
on the [Releases](../../releases) page, built by CI for: Linux x64 (CPU and Vulkan),
macOS arm64 (Metal), and Windows x64 (Vulkan). Each archive holds `zonos2-cli`,
`zonos2-server` (+ the `web/` UI), `spk-encoder-cli`, `dac-cli`, and `quantize-cli` — no
shared-library install needed. The model GGUFs
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

Each build produces **`zonos2-cli`**, **`zonos2-server`**, **`spk-encoder-cli`**, **`dac-cli`**,
**`quantize-cli`**, and **`zonos2-perplexity`**. CUDA-graph replay (needed for the real-time
decode) is enabled automatically for CUDA builds. The server links two header-only libraries
vendored under `vendor/` (cpp-httplib, nlohmann/json) — no extra install.

## Models (one-time conversion)

### Prebuilt GGUFs (Hugging Face)

Skip the conversion below by pulling the ready-made GGUFs from
[`Zyphra/ZONOS2-GGUF`](https://huggingface.co/Zyphra/ZONOS2-GGUF) — the F16 backbone plus
the DAC and speaker-encoder files (identical to what the converter emits):

```bash
hf download Zyphra/ZONOS2-GGUF zonos2-f16.gguf dac.gguf spk-encoder.gguf --local-dir out
```

Ready-made **F16-spine expert quants** of the backbone are also published — drop-in replacements
for `zonos2-f16.gguf` that pair with the same `dac.gguf` / `spk-encoder.gguf` (quality and
sizes in [Quantization](#quantization)):

```bash
# pick one; Q8_0 is effectively lossless, Q4_K is the smallest that still holds audio quality
hf download Zyphra/ZONOS2-GGUF zonos2-q8_0.gguf --local-dir out   # 8.5 GB
hf download Zyphra/ZONOS2-GGUF zonos2-q6_k.gguf --local-dir out   # 6.8 GB
hf download Zyphra/ZONOS2-GGUF zonos2-q5_k.gguf --local-dir out   # 5.8 GB
hf download Zyphra/ZONOS2-GGUF zonos2-q4_k.gguf --local-dir out   # 4.9 GB

# then use it like any backbone, e.g.
zonos2-cli out/zonos2-q4_k.gguf --tts "Hello." out.wav --dac out/dac.gguf --gpu
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

### Voice cloning (one command)

`zonos2-cli` links the speaker encoder, so `--clone <ref_audio>` encodes the reference in-process
(ffmpeg → ECAPA → [2048] x-vector) and injects it — no separate step:

```bash
zonos2-cli out/zonos2-q8_0.gguf --tts "Cloned voice demo." out.wav \
    --dac out/dac.gguf --clone voice.mp3 --spk-encoder out/spk-encoder.gguf --gpu --seed 1
```

Add `--save-speaker emb.npy` to cache the embedding; later runs can reuse it with `--speaker emb.npy`
and skip re-encoding. `--clone` works the same on `--generate`/`--build-prompt`/`--validate`.

The repo's three bundled reference voices live in `ZONOS2/default_voices/*.mp3`. To precompute an
embedding standalone, `spk-encoder-cli out/spk-encoder.gguf --clone voice.mp3 emb.npy` also accepts
`--wav <24kHz-mono.npy>`, `--raw <f32le>`, or a precomputed `--mel <[T,128].npy>`.

### HTTP server

`zonos2-server` loads the backbone + DAC (+ optional speaker encoder) once and serves an HTTP
API that mirrors the reference [`../ZONOS2`](https://huggingface.co/Zyphra/ZONOS2) FastAPI
server, including **low-latency streaming** and **in-process voice cloning**:

```bash
zonos2-server out/zonos2-f16.gguf --dac out/dac.gguf --spk out/spk-encoder.gguf \
    --host 0.0.0.0 --port 1919 --gpu
# then open http://localhost:1919/  for the browser UI
```

| route | method | description |
|---|---|---|
| `/tts/generate` | POST | JSON → **streaming float32 PCM** (`stream:true`, default) or buffered (`format:"wav"`) |
| `/v1/audio/speech` | POST | OpenAI-compatible (`input`, `response_format` = `pcm` streams / `wav` buffers) |
| `/tts/capabilities` | GET | feature flags for the loaded model |
| `/tts/speakers` | GET/POST | list / cache a session speaker (audio upload or `.npy` embedding) |
| `/tts/speakers/{id}/preview` | GET | cached reference audio (WAV) |
| `/v1/models`, `/v1`, `/health` | GET | status / model list |
| `/` | GET | bundled `web/tts_ui.html` |

```bash
# stream raw float32 PCM @ 44.1 kHz (chunked) and play it
curl -sN localhost:1919/tts/generate -d '{"text":"Streaming hello.","seed":1}' | aplay -f FLOAT_LE -r 44100 -c 1
# buffered WAV
curl -s localhost:1919/tts/generate -d '{"text":"Hi.","stream":false,"format":"wav","seed":1}' -o out.wav
# clone a voice by uploading reference audio (base64), per request or cached for the session
curl -s localhost:1919/tts/generate -d "{\"text\":\"Cloned.\",\"seed\":1,\"clean_speaker_background\":true,\
\"speaker_audio_base64\":\"$(base64 -w0 voice.mp3)\"}" -o cloned.pcm
```

The JSON body accepts the reference fields: `text`, sampling (`temperature`, `topk`, `top_p`,
`min_p`, `seed`, `repetition_{window,penalty,codebooks}`, `max_tokens`), conditioning
(`speaking_rate_enabled`/`speaking_rate_bucket`, `quality_enabled`/`quality_buckets`,
`clean_speaker_background`, `accurate_mode`), speaker (`speaker_audio_base64`,
`speaker_embedding_base64` for a `.npy`, `speaker_embedding_id` for a session-cached one),
`fade_out_ms`, `stream`, `format`. Audio uploads are decoded with `ffmpeg` (must be on PATH).

Server flags: `--gpu`/`--cpu`, `--spk <encoder.gguf>` (enables audio-upload cloning),
`--max N` (frame ceiling, default 2000 ≈ 23 s), `--stream-block`/`--stream-context` (streaming
granularity / conv-context frames, ≥16 is seam-free), `--dac-cpu` (run the DAC on CPU under
`--gpu`), `--ui <path>`. One synthesis runs at a time (the model isn't thread-safe); concurrent
requests queue.

> **Differences from the Python server (documented honestly via `/tts/capabilities`):**
> the C++ tokenizer is **byte-level**, so `language` is accepted-but-ignored and
> `text_normalization` is unsupported (`text_normalization_enabled:false`); speaker **blending**
> and a default-voices directory are not implemented. `/tts/generate` (buffered) is **bit-identical
> to `zonos2-cli --tts`** for matching params; streamed audio matches the buffered decode to
> ≈−82 dB on GPU (block-vs-full-decode float variance — bit-identical on CPU or with `--dac-cpu`).
> Output is deterministic for fixed `(text, seed, sampling, max_tokens)`; changing `max_tokens`
> perturbs audio at the float level (it sizes the KV window, which reorders flash-attention sums).

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
| `--speaker emb.npy [--speaker-pos P]` | inject a precomputed voice embedding (default position 0) |
| `--clone ref.{mp3,wav,…} --spk-encoder spk-encoder.gguf` | one-command clone: encode reference in-process (add `--save-speaker emb.npy` to cache) |
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
- **Q8_0 (full)** — bulk 2-D/3-D matrices at Q8_0, 1-D at F32, and the quant-sensitive tensors
  (embedding tables, output head, all router weights) bumped to F16. 7.7 GB, +41 MB over
  pure Q8_0, strictly better against golden; CUDA graphs still replay. A solid one-file build,
  but the F16-spine expert ladder below is both smaller and higher quality at matched size.
- **Never K-quant the whole backbone.** Produced by `quantize-cli` via `ggml_quantize_chunk` (the
  pure-Python converter can't emit K-quants), but sub-8-bit weights on the attention/dense-FFN
  spine perturb the residual just enough to flip the MoE router's top-k expert choice, and the
  output then decorrelates — full Q4_K measures KL-divergence **6.7** / top-1 **5%** vs F16,
  despite the quantizer itself being numerically correct.
- **F16 spine + K-quant experts** (recommended) — `--experts-only --spine-f16` keeps the entire
  spine at F16 and applies the K-quant only to the MoE expert stacks (most of the weights, but
  the bits they tolerate). This is the published HF ladder (Q4_K … Q8_0); see below.

### Making quants from the F16 GGUF

`quantize-cli` requantizes the F16 backbone to any ggml quant type — no checkpoint or Python
needed, so it runs straight off the HF download:

```bash
quantize-cli out/zonos2-f16.gguf out/zonos2-q8_0.gguf q8_0
quantize-cli out/zonos2-f16.gguf out/zonos2-q4_k.gguf q4_k --experts-only --spine-f16
# types: q8_0 q4_0 q4_1 q5_0 q5_1 q2_k q3_k q4_k q5_k q6_k iq4_nl iq4_xs
```

It mirrors the converter's per-tensor policy: 1-D tensors stay F32, the quant-sensitive set
(output head, token/audio embeddings, MoE routers) is kept one tier above the bulk quant, and
every other matrix takes the requested type — falling back to F16 if its row length isn't
block-aligned. The whole F16 file is loaded into RAM (~15 GB) alongside the output, so size
the machine accordingly. Quantizing from F16 (vs the bf16 checkpoint) is numerically
equivalent — f16 is lossless for these weights, so the result matches the converter's Q8_0 to
within quantizer rounding (≈1 element in 4M off by one LSB).

### Recommended: F16 spine + K-quant experts

The MoE expert stacks (`ffn_{gate,up,down}_exps`) are most of the backbone's weights but
tolerate low bits, as long as the router and the residual feeding it stay clean.
`--experts-only --spine-f16` keeps the **entire spine at F16** (attention, dense FFN, routers,
embeddings, head; 1-D stays F32) and applies the K-quant only to the experts. The F16 spine —
not imatrix calibration — is the dominant quality lever: it roughly halves free-run KLD versus a
Q8_0 spine.

Two metric sets vs the F16 backbone, over a multispeaker free-run corpus (the golden prompt is
useless for ranking — every quant scores ~100% top-1 on it). **KLD/Top-1** track per-frame
logits (`zonos2-perplexity`); **WER** (Qwen3-ASR), **SpkSim**, and **UTMOS** are end-to-end audio:

| backbone | bpw | size | KLD ↓ | Top-1 ↑ | WER ↓ | SpkSim ↑ | UTMOS ↑ |
|---|---|---|---|---|---|---|---|
| F16 (ref) | 16.0 | 15.3 GB | — | — | 2.79 | 66.75 | 4.40 |
| **Q8_0** | 8.50 | 8.5 GB | 0.002 | 96.5% | 2.87 | 66.30 | 4.40 |
| **Q6_K** | 6.56 | 6.8 GB | 0.007 | 92.9% | 3.07 | 66.12 | 4.40 |
| **Q5_K** | 5.50 | 5.8 GB | 0.025 | 86.3% | 2.98 | 66.30 | 4.40 |
| **Q4_K** | 4.50 | 4.9 GB | 0.072 | 76.9% | 3.00 | 64.54 | 4.36 |

Although KLD and Top-1 degrade steadily as the experts shrink, **audio quality holds nearly flat
down to Q4_K** — WER, speaker similarity, and UTMOS stay within eval noise of F16. Q8_0 is the
effectively-lossless default; Q6_K is the sweet spot; **Q4_K (~4.25–4.5 bpw) is the usable floor**.
Below that, 3-bit experts fall off a cliff (Q3_K / IQ3_S drop to ~57–59% top-1) and aren't worth
shipping. At equal bpw, plain K-quants beat the IQ variants (IQ4_XS/NL, IQ3_S) on these experts —
it's the codebook geometry, not the calibration — so prefer K-quant.

### Measuring quant quality (`zonos2-perplexity`)

Teacher-forced perplexity and KL-divergence between a quantized backbone and an F16 reference —
the ZONOS2 analogue of llama.cpp's `perplexity` tool. Each `(frame, codebook)` pair is one
prediction event over the 1026-way audio vocab; position *t*'s logits score frame *t+1*'s
codes. A two-pass base-file workflow keeps one model resident at a time, so the F16 reference is
computed once and reused for every quant:

```bash
# 1) write reference distributions from the F16 backbone. A single golden prompt can't rank
#    quants (every quant scores ~100% top-1 on it); use a multispeaker free-run corpus instead.
#    scripts/gen_kld_corpus.sh synthesizes voices + teacher-forced id traces into a manifest:
scripts/gen_kld_corpus.sh   # writes out/kldcorp/*.npy + out/kld.manifest
zonos2-perplexity out/zonos2-f16.gguf --kl-divergence-base out/ref-multi.kld.bin --manifest out/kld.manifest --gpu
# 2) score any quant against that base — prints PPL, KLD mean/median/p99, top-1, per-codebook
zonos2-perplexity out/zonos2-q4_k.gguf --kl-divergence out/ref-multi.kld.bin --gpu
# plain perplexity, no reference needed
zonos2-perplexity out/zonos2-f16.gguf --perplexity out/golden/input_ids.npy
```

The corpus is any set of `[n, n_codebooks+1]` input-id `.npy` files (from `zonos2-cli
--build-prompt` or `--dump-ids`, or a real prompt's `input_ids.npy`) — pass them directly, or via
a `--manifest` that also attaches a speaker `.npy` per trace to exercise the cloned-speaker routes.
The base file embeds the input ids and the reference log-probs, so pass 2 needs only the base and
the quant model.

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
