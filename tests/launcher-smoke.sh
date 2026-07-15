#!/usr/bin/env bash
# Exercise scripts/start-zonos2.sh against the tiny committed fixtures: the
# download-skip path (models already present), the download path (fetch from a
# local http.server), the optional-spk-encoder warning, and trap teardown.
#
# Uses SIGTERM (not SIGINT) to test cleanup: background non-interactive shells
# have SIGINT ignored on entry, so its trap can never fire here — interactively
# (the real double-click case) Ctrl-C works through the same trap.
#
# Usage:  bash tests/launcher-smoke.sh [server-bin]
#   binary defaults to auto-discovery under build*/.  Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAUNCHER="scripts/start-zonos2.sh"
MODEL="tests/fixtures/dummy-zonos2.gguf"
DAC="tests/fixtures/dummy-dac.gguf"

find_bin() {
  if [ -n "${1:-}" ]; then echo "$1"; return; fi
  find build* \( -name zonos2-server -o -name zonos2-server.exe \) -type f 2>/dev/null | head -1
}
SRV="$(find_bin "${1:-}")"
[ -x "$SRV" ]     || { echo "FAIL: zonos2-server not found (pass it as \$1)"; exit 1; }
[ -f "$MODEL" ]   || { echo "FAIL: missing $MODEL"; exit 1; }
[ -f "$DAC" ]     || { echo "FAIL: missing $DAC"; exit 1; }
command -v curl    >/dev/null 2>&1 || { echo "SKIP: curl not available"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "SKIP: python3 not available"; exit 0; }

WORK="$(mktemp -d)"
LNCH_PID=""
HTTP_PID=""
cleanup() {
  local rc=$?
  [ -n "$LNCH_PID" ] && { kill "$LNCH_PID" 2>/dev/null || true; wait "$LNCH_PID" 2>/dev/null || true; }
  [ -n "$HTTP_PID" ] && { kill "$HTTP_PID" 2>/dev/null || true; wait "$HTTP_PID" 2>/dev/null || true; }
  rm -rf "$WORK"
  exit "$rc"
}
trap cleanup EXIT
pass() { echo "  ok: $1"; }
fail() { echo "FAIL: $1"; [ -f "$WORK/launch.log" ] && { echo "--- launcher log ---"; cat "$WORK/launch.log"; }; exit 1; }
is_riff() { [ "$(head -c4 "$1" 2>/dev/null)" = "RIFF" ]; }

PORT="${ZONOS2_TEST_PORT:-19188}"
BASE="http://127.0.0.1:$PORT"

# run_launcher <base_url>: start the launcher against $WORK/models, wait for /health.
run_launcher() {
  ZONOS2_MODEL_DIR="$WORK/models" ZONOS2_SERVER_BIN="$SRV" \
  ZONOS2_PORT="$PORT" ZONOS2_BASE_URL="$1" \
  ZONOS2_ASSUME_YES=1 ZONOS2_NO_BROWSER=1 \
    bash "$LAUNCHER" -- --max 16 >"$WORK/launch.log" 2>&1 &
  LNCH_PID=$!
  local up=0
  for _ in $(seq 1 150); do
    if curl -fsS "$BASE/health" >/dev/null 2>&1; then up=1; break; fi
    kill -0 "$LNCH_PID" 2>/dev/null || fail "launcher exited before the server came up"
    sleep 0.2
  done
  [ "$up" = 1 ] || fail "server did not answer /health in time"
}

# stop_launcher: TERM the launcher, assert its trap takes the server down.
stop_launcher() {
  kill -TERM "$LNCH_PID" 2>/dev/null || true
  wait "$LNCH_PID" 2>/dev/null || true
  LNCH_PID=""
  local down=0
  for _ in $(seq 1 25); do
    if ! curl -fsS "$BASE/health" >/dev/null 2>&1; then down=1; break; fi
    sleep 0.2
  done
  [ "$down" = 1 ] || fail "server still answering after the launcher was stopped"
}

echo "== launcher: download-skip path =="
mkdir -p "$WORK/models"
cp "$MODEL" "$WORK/models/zonos2-q6_k.gguf"
cp "$DAC"   "$WORK/models/dac.gguf"
# Poisoned base URL: backbone+dac are present so no fatal download may happen
# (the missing spk-encoder.gguf fails fast and must only warn).
run_launcher "http://127.0.0.1:9"
# the launcher's own readiness poll may print a beat after ours succeeds
ready=0
for _ in $(seq 1 25); do
  if grep -q "zonos2: ready" "$WORK/launch.log"; then ready=1; break; fi
  sleep 0.2
done
[ "$ready" = 1 ] || fail "no ready line in launcher output"
pass "models present -> server up without downloads"
grep -q "warning: failed to download spk-encoder.gguf" "$WORK/launch.log" \
  || fail "missing spk-encoder.gguf did not produce the non-fatal warning"
pass "spk-encoder download failure is non-fatal"
curl -fsS "$BASE/tts/generate" \
  -d '{"text":"launcher smoke","stream":false,"format":"wav","seed":1,"max_tokens":16}' \
  -o "$WORK/smoke.wav" || fail "/tts/generate request failed"
is_riff "$WORK/smoke.wav" || fail "/tts/generate did not return a RIFF WAV"
pass "/tts/generate -> WAV"
stop_launcher
pass "TERM tears the server down"

echo "== launcher: ffmpeg auto-download =="
# Serve a fake single-binary ffmpeg and force the download (ZONOS2_SKIP_PATH_FFMPEG bypasses
# the system-ffmpeg check that CI/dev boxes would otherwise satisfy). Models already present,
# so only ffmpeg is fetched.
FF_PORT="${ZONOS2_TEST_FF_PORT:-18124}"
mkdir -p "$WORK/ffserve"
printf '#!/bin/sh\necho fake-ffmpeg "$@"\n' > "$WORK/ffserve/ffmpeg"
python3 -m http.server "$FF_PORT" --bind 127.0.0.1 --directory "$WORK/ffserve" >/dev/null 2>&1 &
HTTP_PID=$!
for _ in $(seq 1 50); do curl -fsS "http://127.0.0.1:$FF_PORT/" >/dev/null 2>&1 && break; sleep 0.2; done
rm -rf "$WORK/models/bin"
export ZONOS2_SKIP_PATH_FFMPEG=1 ZONOS2_FFMPEG_URL="http://127.0.0.1:$FF_PORT/ffmpeg"
run_launcher "http://127.0.0.1:9"
[ -x "$WORK/models/bin/ffmpeg" ] || fail "ffmpeg was not downloaded to models/bin"
pass "ffmpeg auto-downloaded to models/bin"
grep -q "downloading ffmpeg" "$WORK/launch.log" || fail "launcher did not announce the ffmpeg download"
pass "ffmpeg download announced"
stop_launcher
unset ZONOS2_SKIP_PATH_FFMPEG ZONOS2_FFMPEG_URL
kill "$HTTP_PID" 2>/dev/null || true; wait "$HTTP_PID" 2>/dev/null || true; HTTP_PID=""

echo "== launcher: download path =="
HTTP_PORT="${ZONOS2_TEST_HTTP_PORT:-18123}"
mkdir -p "$WORK/serve"
cp "$MODEL" "$WORK/serve/zonos2-q6_k.gguf"
cp "$DAC"   "$WORK/serve/dac.gguf"        # deliberately no spk-encoder.gguf (404 -> warn)
python3 -m http.server "$HTTP_PORT" --bind 127.0.0.1 --directory "$WORK/serve" >/dev/null 2>&1 &
HTTP_PID=$!
srv_up=0
for _ in $(seq 1 50); do
  if curl -fsS "http://127.0.0.1:$HTTP_PORT/" >/dev/null 2>&1; then srv_up=1; break; fi
  sleep 0.2
done
[ "$srv_up" = 1 ] || fail "python http.server did not come up"

rm "$WORK/models/dac.gguf"
run_launcher "http://127.0.0.1:$HTTP_PORT"
cmp -s "$WORK/models/dac.gguf" "$DAC" || fail "downloaded dac.gguf is not byte-identical to the fixture"
pass "missing dac.gguf downloaded and byte-identical"
stop_launcher
pass "second run torn down"

echo "All launcher smoke tests passed."
