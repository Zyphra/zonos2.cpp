# zonos2.cpp

<p align="center">
  <img src="./assets/ZONOS2BlogThumbnail.png" alt="ZONOS2" width="750" />
</p>

<div align="center">
  <a href="https://discord.gg/gTW9JwST8q" target="_blank">
    <img src="https://img.shields.io/badge/Join%20Our%20Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

---

zonos2.cpp is a standalone [ggml](https://github.com/ggml-org/ggml)/GGUF C++ port of
[ZONOS2](https://huggingface.co/Zyphra/ZONOS2), Zyphra's ~7.6B-param MoE text-to-speech model.
The **entire pipeline** — ECAPA-TDNN speaker encoder, the MoE language-model backbone, and the
DAC-44kHz vocoder — runs as native C++ linking only `libggml` + `gguf`. **No Python, no PyTorch,
no CUDA-only kernels at inference time.** It runs on **CPU, CUDA, Apple Metal, and Vulkan** from
the same GGUF files, in real time on GPU, and quantizes down to **4.9 GB (Q4_K)** with audio
quality within eval noise of F16.

An inference overview can be seen below.

<p align="center">
  <img src="./assets/zonos2_arlooop_animated.gif" alt="ZONOS2 inference overview" width="750" />
</p>

zonos2.cpp ships a high-performance HTTP server (`zonos2-server`) that mirrors the reference
FastAPI server — low-latency streaming PCM, an OpenAI-compatible `/v1/audio/speech` route,
in-process voice cloning, and a browser UI — plus a CLI (`zonos2-cli`) for offline synthesis.
Every stage is numerically validated against the PyTorch reference — the speaker encoder and DAC
decoder bit-exact, the backbone to cosine ≥ 0.9999.

**For more details and speech samples, check out the [blog](https://www.zyphra.com/our-work/zonos2).**

**A hosted version is available at [cloud.zyphra.com/audio-playground](https://cloud.zyphra.com/audio-playground).**

---

## Quick Start

> **Platform Support**: Linux x64 (CPU / Vulkan), macOS arm64 (Metal), and Windows x64 (Vulkan)
> ship prebuilt. For CUDA (sm_90 / H100 by default), build from source.

### 1. Installation

Grab a self-contained build from the [Releases](../../releases) page (each archive holds
`zonos2-server` + the `web/` UI, `zonos2-cli`, `spk-encoder-cli`, `dac-cli`, and `quantize-cli` —
no shared-library install), or build from source:

```bash
git clone --recurse-submodules https://github.com/Zyphra/zonos2.cpp.git && cd zonos2.cpp

# CPU
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# CUDA (H100 = sm_90; pass -DCMAKE_CUDA_ARCHITECTURES=<n> for other GPUs)
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-cuda -j
```

> **First run on macOS / Windows:** the release binaries are unsigned (built by CI, with no
> code-signing certificate), so the OS may block them the *first* time you run one. This is
> expected — the warning is about the missing signature, not the binary. Building from source
> avoids it entirely.

<details>
<summary>Clearing the OS security warning on the prebuilt binaries</summary>

- **macOS (Gatekeeper)** — you'll see *"…cannot be opened because the developer cannot be
  verified"* (or *"Apple could not verify … is free of malware"* on Sequoia). macOS flags every
  file in a downloaded archive with a quarantine attribute; strip it from the folder you extracted
  into, then run the tools normally:
  ```bash
  xattr -dr com.apple.quarantine /path/to/extracted-folder
  ```
  Alternatively, after the first blocked launch, allow it via **System Settings → Privacy &
  Security → Open Anyway**.

- **Windows (SmartScreen / Defender)** — *"Windows protected your PC"* appears the first time you
  launch `zonos2-server.exe` / `zonos2-cli.exe`. Click **More info → Run anyway**. To clear the
  block before extracting, unblock the downloaded archive (right-click → **Properties → Unblock →
  OK**, or in PowerShell):
  ```powershell
  Unblock-File .\zonos2-windows-x64-vulkan.tar.gz
  ```
  If Microsoft Defender quarantines the unrecognized `.exe`, restore it from Protection History
  or add a Defender exclusion for the folder.

- **Linux** — no signing prompt. If a binary won't start, it usually just needs the execute bit:
  ```bash
  chmod +x zonos2-cli zonos2-server
  ```

</details>

### 2. Install ffmpeg (required for voice cloning)

Voice cloning decodes reference audio by shelling out to `ffmpeg` — the server's `--spk`
upload route, `zonos2-cli --clone`, and `spk-encoder-cli --clone` all need it on your `PATH`.
Plain TTS works without it.

```bash
# macOS
brew install ffmpeg
# Debian / Ubuntu
sudo apt install ffmpeg
# Windows
winget install ffmpeg    # or: choco install ffmpeg / scoop install ffmpeg
```

### 3. Get the Models

Pull the ready-made GGUFs from [`Zyphra/ZONOS2-GGUF`](https://huggingface.co/Zyphra/ZONOS2-GGUF):

```bash
hf download Zyphra/ZONOS2-GGUF dac.gguf spk-encoder.gguf --local-dir out
# pick a backbone: Q8_0 is effectively lossless, Q4_K is the smallest that holds quality
hf download Zyphra/ZONOS2-GGUF zonos2-q8_0.gguf --local-dir out   # 8.5 GB
hf download Zyphra/ZONOS2-GGUF zonos2-q4_k.gguf --local-dir out   # 4.9 GB
```

(You can also convert the original checkpoints and make any quant yourself — see
[docs/INTERNALS.md](docs/INTERNALS.md).)

### 4. Launch the TTS Server

```bash
zonos2-server out/zonos2-q8_0.gguf --dac out/dac.gguf --spk out/spk-encoder.gguf \
    --host 0.0.0.0 --port 1919 --gpu --tts-default-voices-dir default_voices
```

The server starts on `http://localhost:1919` by default. It loads the backbone + DAC (+ speaker
encoder) once and serves the reference TTS API: streaming float32 PCM, the OpenAI-compatible
`/v1/audio/speech` route, audio-upload voice cloning, cached/default speakers, and SLERP speaker
blending. `--spk` enables cloning from uploaded reference audio and default voice audio files;
drop `--gpu` for CPU. Emotion direction files in `./emotion_directions/` are autoloaded when
present; use `--tts-emotion-directions-dir <dir>` to point elsewhere or pass an empty directory
string to disable emotion controls. Default voices are scanned from `default_voices/` by default;
use `--tts-default-voices-dir <dir>` to point elsewhere or pass an empty string to disable them.

Text normalization is available through the reference NeMo/Pynini helper when launched with a
Python environment that has `pynini` installed:

```bash
zonos2-server out/zonos2-q8_0.gguf --dac out/dac.gguf --spk out/spk-encoder.gguf \
    --gpu --text-normalizer-python .venv-tts-norm/bin/python
```

Without `--text-normalizer-python`, the server keeps the pure byte-level tokenizer path and reports
`text_normalization_enabled:false`.

### 5. Generate Speech

**curl:**

```bash
# stream raw float32 PCM @ 44.1 kHz
curl -X POST http://localhost:1919/tts/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "stream": true, "seed": 1}' \
  --output output.pcm

# convert to WAV
ffmpeg -f f32le -ar 44100 -ac 1 -i output.pcm output.wav

# or get a buffered WAV directly
curl -s http://localhost:1919/tts/generate \
  -d '{"text":"Hi.","stream":false,"format":"wav","seed":1}' -o out.wav
```

**Web UI:** Open `http://localhost:1919/` in your browser.

## Emotion Control

Emotion control nudges a cloned speaker voice with shipped direction vectors: named sliders
(`happy`, `sad`, `angry`, `surprised`) plus `valence` and `arousal` axes. It requires a speaker
embedding from an uploaded/cached voice or CLI `--speaker` / `--clone`.

```bash
curl -X POST http://localhost:1919/tts/generate \
  -H "Content-Type: application/json" \
  -d '{
        "text": "I cannot believe you did that!",
        "speaker_embedding_id": "spk_...",
        "emotion_enabled": true,
        "emotion_sliders": {"happy": 1.0},
        "accurate_mode": false,
        "emotion_cfg_scale": 1.5,
        "stream": true
      }' \
  --output happy.pcm
```

`GET /tts/capabilities` reports `emotion_enabled`, `emotion_names`, `emotion_axes`, and
`emotion_calibrated`. `emotion_strength` is a multiplier on the loaded calibration when
`calibration.json` is present.

## CLI (offline inference)

You can also synthesize directly from the command line, without starting a server. One command
turns text into a 44.1 kHz WAV:

```bash
zonos2-cli out/zonos2-q8_0.gguf --tts "Hello, world." out.wav \
    --dac out/dac.gguf --gpu --seed 1
```

Clone a voice in-process by pointing `--clone` at any reference audio:

```bash
zonos2-cli out/zonos2-q8_0.gguf --tts "Cloned voice demo." out.wav \
    --dac out/dac.gguf --clone voice.mp3 --spk-encoder out/spk-encoder.gguf --gpu --seed 1
```

CLI emotion example:

```bash
zonos2-cli out/zonos2-q8_0.gguf --tts "That was incredible." out.wav \
    --dac out/dac.gguf --speaker voice.npy --emotion happy=1 \
    --emotion-strength 1 --emotion-cfg-scale 1.5 --gpu --seed 1
```

## Under the Hood

zonos2.cpp is a faithful, numerically-validated port. For the full conversion flow, the
quantization ladder (F16 spine + K-quant experts, with WER / SpkSim / UTMOS evals), performance
(H100 and Apple Silicon), the validation harness, and the architecture spec, see
**[docs/INTERNALS.md](docs/INTERNALS.md)**.

## Credits

- [ggml](https://github.com/ggml-org/ggml) — the tensor library / GGUF format.
- [Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2) — the original model and PyTorch reference
  this port mirrors and is validated against.
- [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) — the DAC vocoder.

## Citation

If you find this model useful in an academic context please cite as:

```
@misc{zyphra2025zonos,
  title     = {Zonos V2 Technical Report},
  author    = {Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, Beren Millidge},
  year      = {2026},
}
```
