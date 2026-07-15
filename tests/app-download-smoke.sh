#!/usr/bin/env bash
# Exercise zonos2-app's model download via its headless self-test seam
# (ZONOS2_APP_SELFTEST_DOWNLOAD=<quant>): fetch backbone + dac + spk-encoder from a local
# http.server into <config>/models and assert the files are byte-identical. The GUI/JS path
# that drives this in the real app isn't headless-drivable, so the seam stands in for it.
#
# Usage:  bash tests/app-download-smoke.sh [app-bin]
#   binary defaults to auto-discovery under build*/.  Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

find_bin() {
  if [ -n "${1:-}" ]; then echo "$1"; return; fi
  find build* \( -name zonos2-app -o -name zonos2-app.exe \) -type f 2>/dev/null | head -1
}
APP="$(find_bin "${1:-}")"
[ -x "$APP" ]      || { echo "FAIL: zonos2-app not found (pass it as \$1)"; exit 1; }
command -v curl    >/dev/null 2>&1 || { echo "SKIP: curl not available"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "SKIP: python3 not available"; exit 0; }
command -v cmp     >/dev/null 2>&1 || { echo "SKIP: cmp not available"; exit 0; }

WORK="$(mktemp -d)"
HTTP_PID=""
cleanup() {
  local rc=$?
  [ -n "$HTTP_PID" ] && { kill "$HTTP_PID" 2>/dev/null || true; wait "$HTTP_PID" 2>/dev/null || true; }
  rm -rf "$WORK"
  exit "$rc"
}
trap cleanup EXIT
pass() { echo "  ok: $1"; }
fail() { echo "FAIL: $1"; [ -f "$WORK/app.log" ] && { echo "--- app log ---"; cat "$WORK/app.log"; }; exit 1; }

PORT="${ZONOS2_TEST_APP_HTTP_PORT:-18989}"

echo "== app: model download self-test =="
# Distinctive random payloads. The backbone is >32 MB so the segmented (parallel-range) path
# is exercised; dac/spk stay under the threshold and take the single-stream path. tests/range_server.py
# answers Range requests with 206 (the stdlib http.server ignores Range), emulating HF's CDN.
mkdir -p "$WORK/serve"
head -c 40000000 /dev/urandom > "$WORK/serve/zonos2-q6_k.gguf"
head -c   500000 /dev/urandom > "$WORK/serve/dac.gguf"
head -c    40000 /dev/urandom > "$WORK/serve/spk-encoder.gguf"
python3 "$ROOT/tests/range_server.py" "$PORT" "$WORK/serve" >/dev/null 2>&1 &
HTTP_PID=$!
up=0
for _ in $(seq 1 50); do
  if curl -fsI "http://127.0.0.1:$PORT/dac.gguf" >/dev/null 2>&1; then up=1; break; fi
  sleep 0.2
done
[ "$up" = 1 ] || fail "python http.server did not come up"

# The app writes into <config>/models; point the config dir (XDG on Linux) at $WORK.
export XDG_CONFIG_HOME="$WORK/config"
MODELS="$XDG_CONFIG_HOME/zonos2/models"
rm -rf "$MODELS"

ZONOS2_BASE_URL="http://127.0.0.1:$PORT" ZONOS2_APP_SELFTEST_DOWNLOAD=q6_k \
  "$APP" >"$WORK/app.log" 2>&1 || fail "self-test exited nonzero"
grep -q "selftest-download: ok=1" "$WORK/app.log" || fail "self-test did not report ok=1"
pass "self-test downloaded and reported success"
grep -q "zonos2-q6_k.gguf fetched with .* parallel connections" "$WORK/app.log" \
  || fail "backbone did not use the segmented (parallel) download path"
pass "backbone fetched via segmented parallel download"

for f in zonos2-q6_k.gguf dac.gguf spk-encoder.gguf; do
  cmp -s "$WORK/serve/$f" "$MODELS/$f" || fail "$f missing or not byte-identical"
  pass "$f byte-identical"
done
[ -e "$MODELS/zonos2-q6_k.gguf.part" ] && fail "leftover .part file after download"
pass "no leftover .part files"

echo "All app download smoke tests passed."
