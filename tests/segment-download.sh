#!/usr/bin/env bash
# Exercise the launcher's segmented (parallel-range) download in isolation: source
# start-zonos2.sh with ZONOS2_LIB_ONLY=1 (loads the functions, skips the main flow), serve a
# >32 MB file from a local threaded http.server, and assert download_segmented reassembles it
# byte-identically. Also checks the small-file threshold (returns 1 → caller falls back).
#
# Usage:  bash tests/segment-download.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v curl    >/dev/null 2>&1 || { echo "SKIP: curl not available"; exit 0; }
command -v python3 >/dev/null 2>&1 || { echo "SKIP: python3 not available"; exit 0; }
command -v cmp     >/dev/null 2>&1 || { echo "SKIP: cmp not available"; exit 0; }

# Load the launcher's functions without running it.
ZONOS2_LIB_ONLY=1 . scripts/start-zonos2.sh
# model_size_gb (called for the progress line) only knows the real names; stub it for fixtures.
model_size_gb() { echo 0.04; }

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
fail() { echo "FAIL: $1"; exit 1; }

PORT="${ZONOS2_TEST_SEG_PORT:-18991}"
mkdir -p "$WORK/serve"
head -c 40000000 /dev/urandom > "$WORK/serve/big.bin"     # > 32 MB -> segmented
head -c   100000 /dev/urandom > "$WORK/serve/small.bin"   # < 32 MB -> threshold returns 1
# range_server.py answers Range with 206 (stdlib http.server ignores Range), emulating HF's CDN.
python3 "$ROOT/tests/range_server.py" "$PORT" "$WORK/serve" >/dev/null 2>&1 &
HTTP_PID=$!
for _ in $(seq 1 50); do curl -fsI "http://127.0.0.1:$PORT/big.bin" >/dev/null 2>&1 && break; sleep 0.2; done

echo "== segmented reassembly (8 connections) =="
ZONOS2_DL_CONNECTIONS=8 \
  download_segmented "big.bin" "http://127.0.0.1:$PORT/big.bin" "$WORK/big.out" >"$WORK/seg.log" 2>&1 \
  || fail "download_segmented returned failure on a >32MB file"
cmp -s "$WORK/serve/big.bin" "$WORK/big.out" || fail "reassembled file is not byte-identical"
pass "40 MB file reassembled byte-identical"
grep -q "8 connections" "$WORK/seg.log" || fail "did not announce the parallel connections"
pass "announced parallel connections"
ls "$WORK"/big.out.part* >/dev/null 2>&1 && fail "leftover .partN segments" || pass "no leftover segments"

echo "== small file skips segmentation =="
if ZONOS2_DL_CONNECTIONS=8 \
     download_segmented "small.bin" "http://127.0.0.1:$PORT/small.bin" "$WORK/small.out" >/dev/null 2>&1; then
  fail "small file should have returned non-zero (below the 32MB threshold)"
fi
[ -e "$WORK/small.out" ] && fail "small file should not have been written by download_segmented"
pass "sub-threshold file declined (caller falls back to single stream)"

echo "== single connection disables segmentation =="
if ZONOS2_DL_CONNECTIONS=1 \
     download_segmented "big.bin" "http://127.0.0.1:$PORT/big.bin" "$WORK/one.out" >/dev/null 2>&1; then
  fail "ZONOS2_DL_CONNECTIONS=1 should decline segmentation"
fi
pass "single-connection setting declines segmentation"

echo "All segmented download tests passed."
