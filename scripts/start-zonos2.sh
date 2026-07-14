#!/usr/bin/env bash
# start-zonos2 — download the models if missing, launch zonos2-server, open the browser.
# Double-click friendly: on macOS this file also ships as start-zonos2.command.
# Must stay bash-3.2 compatible (stock macOS bash).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU_DEFAULT=cpu   # CI stamps this to "gpu" in the vulkan/metal release archives
ZONOS2_QUANT="${ZONOS2_QUANT:-q6_k}"
ZONOS2_MODEL_DIR="${ZONOS2_MODEL_DIR:-$SCRIPT_DIR/models}"
ZONOS2_BASE_URL="${ZONOS2_BASE_URL:-https://huggingface.co/Zyphra/ZONOS2-GGUF/resolve/main}"
ZONOS2_HOST="${ZONOS2_HOST:-127.0.0.1}"
ZONOS2_PORT="${ZONOS2_PORT:-1919}"
ZONOS2_SERVER_BIN="${ZONOS2_SERVER_BIN:-$SCRIPT_DIR/zonos2-server}"
ASSUME_YES="${ZONOS2_ASSUME_YES:-0}"
NO_BROWSER="${ZONOS2_NO_BROWSER:-0}"
GPU_MODE="$GPU_DEFAULT"
EXTRA_ARGS=()

usage() {
    cat <<EOF
usage: $0 [options] [-- <extra zonos2-server args>]
  --quant Q      model quant to fetch/serve (q4_k 4.9GB, q5_k 5.8GB, q6_k 6.8GB,
                 q8_0 8.5GB, f16 15.3GB; default $ZONOS2_QUANT)
  --cpu | --gpu  backend (archive default: $GPU_DEFAULT)
  --host H       bind address (default $ZONOS2_HOST)
  --port P       port (default $ZONOS2_PORT)
  -y, --yes      don't ask before downloading
  --no-browser   don't open the web UI
env overrides: ZONOS2_QUANT ZONOS2_MODEL_DIR ZONOS2_BASE_URL ZONOS2_HOST ZONOS2_PORT
               ZONOS2_SERVER_BIN ZONOS2_ASSUME_YES ZONOS2_NO_BROWSER
EOF
}

die() { echo "start-zonos2: error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --quant)      [ $# -ge 2 ] || die "--quant needs a value"; ZONOS2_QUANT="$2"; shift 2 ;;
        --cpu)        GPU_MODE=cpu; shift ;;
        --gpu)        GPU_MODE=gpu; shift ;;
        --host)       [ $# -ge 2 ] || die "--host needs a value"; ZONOS2_HOST="$2"; shift 2 ;;
        --port)       [ $# -ge 2 ] || die "--port needs a value"; ZONOS2_PORT="$2"; shift 2 ;;
        -y|--yes)     ASSUME_YES=1; shift ;;
        --no-browser) NO_BROWSER=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; EXTRA_ARGS=("$@"); break ;;
        *)            usage; die "unknown option '$1' (pass server flags after --)" ;;
    esac
done

[ -x "$ZONOS2_SERVER_BIN" ] || die "zonos2-server not found at $ZONOS2_SERVER_BIN (run this script from the extracted release archive, or set ZONOS2_SERVER_BIN)"

model_size_gb() {
    case "$1" in
        zonos2-f16.gguf)  echo 15.3 ;;
        zonos2-q8_0.gguf) echo 8.5 ;;
        zonos2-q6_k.gguf) echo 6.8 ;;
        zonos2-q5_k.gguf) echo 5.8 ;;
        zonos2-q4_k.gguf) echo 4.9 ;;
        dac.gguf)         echo 0.25 ;;
        spk-encoder.gguf) echo 0.02 ;;
        *)                echo "?" ;;
    esac
}

# curl -C - --fail exits 33 when the .part is already complete (HTTP 416): restart clean once.
download_one() {
    name="$1"
    url="$ZONOS2_BASE_URL/$name"
    dst="$ZONOS2_MODEL_DIR/$name"
    echo "downloading $name ($(model_size_gb "$name") GB) ..."
    rc=0
    curl -L --fail --retry 3 -C - --progress-bar -o "$dst.part" "$url" || rc=$?
    if [ "$rc" = 33 ]; then
        rm -f "$dst.part"
        rc=0
        curl -L --fail --retry 3 --progress-bar -o "$dst.part" "$url" || rc=$?
    fi
    [ "$rc" = 0 ] || return "$rc"
    mv "$dst.part" "$dst"
}

BACKBONE="zonos2-$ZONOS2_QUANT.gguf"
MISSING=()
for f in "$BACKBONE" dac.gguf spk-encoder.gguf; do
    [ -f "$ZONOS2_MODEL_DIR/$f" ] || MISSING+=("$f")
done

if [ "${#MISSING[@]}" -gt 0 ]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to download models — install it (e.g. 'sudo apt install curl' / 'brew install curl'), or place the .gguf files in $ZONOS2_MODEL_DIR yourself (see README)"
    echo "The following models are missing from $ZONOS2_MODEL_DIR and will be downloaded:"
    for f in "${MISSING[@]}"; do
        echo "  $f  ($(model_size_gb "$f") GB)"
    done
    echo "  from $ZONOS2_BASE_URL"
    if [ "$ASSUME_YES" != 1 ] && [ -t 0 ]; then
        printf "Download now? [Y/n] "
        read -r ans
        case "$ans" in ""|y*|Y*) ;; *) echo "aborted."; exit 1 ;; esac
    fi
    mkdir -p "$ZONOS2_MODEL_DIR" || die "cannot create $ZONOS2_MODEL_DIR (set ZONOS2_MODEL_DIR to a writable location)"
    for f in "${MISSING[@]}"; do
        if ! download_one "$f"; then
            if [ "$f" = spk-encoder.gguf ]; then
                echo "start-zonos2: warning: failed to download $f — continuing without voice-clone upload support" >&2
            else
                die "failed to download $f from $ZONOS2_BASE_URL/$f"
            fi
        fi
    done
fi

# The UI, default voices, and emotion directions ship next to this script in release
# archives; in a source checkout the script lives in scripts/ with them one level up.
# Pass them explicitly because the server defaults are cwd-relative and a
# double-clicked .command runs from $HOME.
sibling() {  # $1 = name: echo the archive-layout path, falling back to the repo layout
    if [ -e "$SCRIPT_DIR/$1" ]; then echo "$SCRIPT_DIR/$1"; else echo "$SCRIPT_DIR/../$1"; fi
}

args=("$ZONOS2_MODEL_DIR/$BACKBONE"
      --dac "$ZONOS2_MODEL_DIR/dac.gguf"
      --host "$ZONOS2_HOST" --port "$ZONOS2_PORT"
      --ui "$(sibling web/tts_ui.html)")
[ -d "$(sibling default_voices)" ]     && args+=(--tts-default-voices-dir "$(sibling default_voices)")
[ -d "$(sibling emotion_directions)" ] && args+=(--tts-emotion-directions-dir "$(sibling emotion_directions)")
[ -f "$ZONOS2_MODEL_DIR/spk-encoder.gguf" ] && args+=(--spk "$ZONOS2_MODEL_DIR/spk-encoder.gguf")
[ "$GPU_MODE" = gpu ] && args+=(--gpu)

"$ZONOS2_SERVER_BIN" "${args[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} &
SRV_PID=$!
trap 'kill "$SRV_PID" 2>/dev/null || true' INT TERM EXIT

# Poll readiness; bail out early if the server died (its error is already on the terminal).
POLL_HOST="$ZONOS2_HOST"
case "$POLL_HOST" in 0.0.0.0|::) POLL_HOST=127.0.0.1 ;; esac
URL="http://$POLL_HOST:$ZONOS2_PORT/"
ready=0
i=0
while [ $i -lt 240 ]; do
    kill -0 "$SRV_PID" 2>/dev/null || die "server exited during startup"
    if curl -fsS "${URL}health" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.5
    i=$((i + 1))
done
[ "$ready" = 1 ] || die "server did not become ready at $URL"

echo "zonos2: ready -> $URL"
if [ "$NO_BROWSER" != 1 ]; then
    case "$(uname -s)" in
        Darwin) open "$URL" 2>/dev/null || true ;;
        *)      command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL" 2>/dev/null || true ;;
    esac
fi

echo "Press Ctrl-C (or close this window) to stop the server."
wait "$SRV_PID"
