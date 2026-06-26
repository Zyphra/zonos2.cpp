#!/usr/bin/env bash
# Exercise the command flows shown in README.md / docs/INTERNALS.md against the tiny
# committed fixtures (random-weight backbone + DAC), so the documented commands are
# proven to actually run on CI — not just the usage string.
#
# Audio is gibberish by design; we only assert each command runs and emits the right
# kind of artifact (codes .npy, RIFF WAV, float32 PCM). Everything runs on the CPU
# backend (no --gpu) for determinism, matching the existing smoke test.
#
# Usage:  bash tests/readme-commands.sh [cli-bin] [server-bin]
#   binaries default to auto-discovery under build/.  Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="tests/fixtures/dummy-zonos2.gguf"
DAC="tests/fixtures/dummy-dac.gguf"

find_bin() {  # $1 = base name; honor an explicit path arg, else search build*/
  if [ -n "${2:-}" ]; then echo "$2"; return; fi
  find build* \( -name "$1" -o -name "$1.exe" \) -type f 2>/dev/null | head -1
}
CLI="$(find_bin zonos2-cli "${1:-}")"
SRV="$(find_bin zonos2-server "${2:-}")"
[ -x "$CLI" ] || { echo "FAIL: zonos2-cli not found (pass it as \$1)"; exit 1; }
[ -f "$MODEL" ] || { echo "FAIL: missing $MODEL"; exit 1; }
[ -f "$DAC" ]   || { echo "FAIL: missing $DAC"; exit 1; }

WORK="$(mktemp -d)"
SRV_PID=""
cleanup() {  # preserve the script's real exit status across server teardown
  local rc=$?
  [ -n "$SRV_PID" ] && { kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; }
  rm -rf "$WORK"
  exit "$rc"
}
trap cleanup EXIT
pass() { echo "  ok: $1"; }
fail() { echo "FAIL: $1"; exit 1; }
is_riff() { [ "$(head -c4 "$1" 2>/dev/null)" = "RIFF" ]; }

echo "== CLI ($CLI) =="

# build the prompt id matrix (docs/INTERNALS.md "Standalone components")
"$CLI" "$MODEL" --build-prompt "hello prompt" "$WORK/ids.npy" >/dev/null 2>&1 \
  || fail "--build-prompt exited nonzero"
[ -s "$WORK/ids.npy" ] || fail "--build-prompt wrote no ids.npy"
pass "--build-prompt -> ids.npy"

# generate codes from the prompt
"$CLI" "$MODEL" --generate "$WORK/ids.npy" "$WORK/codes.npy" --max 8 --greedy >/dev/null 2>&1 \
  || fail "--generate exited nonzero"
[ -s "$WORK/codes.npy" ] || fail "--generate wrote no codes.npy"
pass "--generate -> codes.npy"

# one-command text -> codes (.npy output, no DAC)
out="$("$CLI" "$MODEL" --tts "ci tts to codes" "$WORK/t.npy" --max 8 --greedy 2>&1)" \
  || { echo "$out"; fail "--tts (codes) exited nonzero"; }
echo "$out" | grep -qE "tts: [0-9]+ frames, eos_frame=" || { echo "$out"; fail "--tts (codes) no generation line"; }
[ -s "$WORK/t.npy" ] || fail "--tts (codes) wrote no .npy"
pass "--tts -> codes .npy"

# one-command text -> WAV, decoding through the DAC (README Quick Start / CLI section)
"$CLI" "$MODEL" --tts "ci tts to wav" "$WORK/t.wav" --dac "$DAC" --max 8 --greedy >/dev/null 2>&1 \
  || fail "--tts --dac (wav) exited nonzero"
is_riff "$WORK/t.wav" || fail "--tts --dac did not write a RIFF WAV"
pass "--tts --dac -> WAV"

# standalone codes -> wav (dac-cli), if present
DACCLI="$(find_bin dac-cli "")"
if [ -x "$DACCLI" ]; then
  "$DACCLI" "$DAC" "$WORK/codes.npy" "$WORK/d.wav" >/dev/null 2>&1 || fail "dac-cli exited nonzero"
  is_riff "$WORK/d.wav" || fail "dac-cli did not write a RIFF WAV"
  pass "dac-cli codes -> WAV"
fi

# ----------------------------------------------------------------- server
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) echo "== server: skipped on Windows =="; exit 0 ;;
esac
if ! command -v curl >/dev/null 2>&1; then echo "== server: skipped (no curl) =="; exit 0; fi
[ -x "$SRV" ] || { echo "== server: skipped (zonos2-server not found) =="; exit 0; }

echo "== server ($SRV) =="
PORT="${ZONOS2_TEST_PORT:-19187}"
BASE="http://127.0.0.1:$PORT"
"$SRV" "$MODEL" --dac "$DAC" --host 127.0.0.1 --port "$PORT" --max 16 >"$WORK/srv.log" 2>&1 &
SRV_PID=$!

up=0
for _ in $(seq 1 100); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then up=1; break; fi
  kill -0 "$SRV_PID" 2>/dev/null || { echo "--- server log ---"; cat "$WORK/srv.log"; fail "server exited before coming up"; }
  sleep 0.2
done
[ "$up" = 1 ] || { echo "--- server log ---"; cat "$WORK/srv.log"; fail "server did not answer /health in time"; }
pass "/health"

curl -fsS "$BASE/tts/capabilities" | grep -q '"n_codebooks"' || fail "/tts/capabilities missing n_codebooks"
pass "/tts/capabilities"

# buffered WAV (README curl example)
curl -fsS "$BASE/tts/generate" \
  -d '{"text":"buffered wav","stream":false,"format":"wav","seed":1,"max_tokens":16}' \
  -o "$WORK/srv.wav" || fail "/tts/generate (buffered wav) request failed"
is_riff "$WORK/srv.wav" || fail "/tts/generate (buffered) did not return a RIFF WAV"
pass "/tts/generate -> buffered WAV"

# streaming float32 PCM (README curl example)
curl -fsS -N "$BASE/tts/generate" \
  -d '{"text":"streaming pcm","stream":true,"seed":1,"max_tokens":16}' \
  -o "$WORK/srv.pcm" || fail "/tts/generate (streaming pcm) request failed"
[ -s "$WORK/srv.pcm" ] || fail "/tts/generate (streaming) returned no PCM"
pass "/tts/generate -> streaming PCM ($(wc -c < "$WORK/srv.pcm") bytes)"

# OpenAI-compatible route
curl -fsS "$BASE/v1/audio/speech" \
  -d '{"input":"openai route","response_format":"wav","seed":1,"max_tokens":16}' \
  -o "$WORK/oai.wav" || fail "/v1/audio/speech request failed"
is_riff "$WORK/oai.wav" || fail "/v1/audio/speech did not return a RIFF WAV"
pass "/v1/audio/speech -> WAV"

echo "All README command tests passed."
