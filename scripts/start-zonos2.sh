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
FFMPEG_URL="${ZONOS2_FFMPEG_URL:-}"   # override with a single-binary URL (used by tests); default is per-OS
FFMPEG_DIR="$ZONOS2_MODEL_DIR/bin"
FFMPEG_BIN="$FFMPEG_DIR/ffmpeg"
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
               ZONOS2_SERVER_BIN ZONOS2_ASSUME_YES ZONOS2_NO_BROWSER ZONOS2_FFMPEG_URL
               ZONOS2_DL_CONNECTIONS (parallel download connections, default 8, max 16)
ffmpeg (voice cloning only) is fetched into $ZONOS2_MODEL_DIR/bin if not already there or
on PATH; basic TTS needs no ffmpeg.
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

# Segmented parallel download: one HEAD resolves the exact size + final CDN URL (HF's resolve/
# endpoint 302s to a Xet CDN that honors Range), then N concurrent range requests are stitched
# back together — the trick hf_transfer uses to beat single-stream HF (~2x+ here). Falls back to
# a single stream (below) on any hiccup. Tunable via ZONOS2_DL_CONNECTIONS (default 8, max 16).
download_segmented() {
    seg_name="$1"; seg_url="$2"; seg_dst="$3"
    conns="${ZONOS2_DL_CONNECTIONS:-8}"
    case "$conns" in ''|*[!0-9]*) conns=8 ;; esac
    [ "$conns" -lt 1 ] && conns=1
    [ "$conns" -gt 16 ] && conns=16
    [ "$conns" -le 1 ] && return 1
    # resolve final URL + exact size from a single redirect-following HEAD
    seg_head=$(curl -sIL "$seg_url" | tr -d '\r') || return 1
    seg_size=$(printf '%s\n' "$seg_head" | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
    seg_eff=$(printf  '%s\n' "$seg_head" | awk 'tolower($1)=="location:"{v=$2} END{print v}')
    [ -n "$seg_eff" ] || seg_eff="$seg_url"
    case "$seg_size" in ''|*[!0-9]*) return 1 ;; esac
    [ "$seg_size" -lt 33554432 ] && return 1   # < 32 MB: not worth segmenting
    seg_len=$((seg_size / conns))
    rm -f "$seg_dst".part[0-9]*
    echo "downloading $seg_name ($(model_size_gb "$seg_name") GB, $conns connections) ..."
    k=0; pids=""
    while [ "$k" -lt "$conns" ]; do
        s=$((k * seg_len))
        if [ "$k" -eq $((conns - 1)) ]; then e=$((seg_size - 1)); else e=$(((k + 1) * seg_len - 1)); fi
        curl -sfL --retry 3 --range "$s-$e" -o "$seg_dst.part$k" "$seg_eff" &
        pids="$pids $!"
        k=$((k + 1))
    done
    seg_ok=1
    for p in $pids; do wait "$p" || seg_ok=0; done
    if [ "$seg_ok" != 1 ]; then rm -f "$seg_dst".part[0-9]*; return 1; fi
    : > "$seg_dst.part"
    k=0
    while [ "$k" -lt "$conns" ]; do
        cat "$seg_dst.part$k" >> "$seg_dst.part" || { rm -f "$seg_dst".part*; return 1; }
        rm -f "$seg_dst.part$k"
        k=$((k + 1))
    done
    if [ "$(wc -c < "$seg_dst.part")" != "$seg_size" ]; then rm -f "$seg_dst.part"; return 1; fi
    mv "$seg_dst.part" "$seg_dst"
}

# curl -C - --fail exits 33 when the .part is already complete (HTTP 416): restart clean once.
download_one() {
    name="$1"
    url="$ZONOS2_BASE_URL/$name"
    dst="$ZONOS2_MODEL_DIR/$name"
    download_segmented "$name" "$url" "$dst" && return 0   # fast path; falls through on any failure
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

ffmpeg_size_mb() { case "$(uname -s)" in Darwin) echo 45 ;; *) echo 108 ;; esac; }

# Fetch a static ffmpeg into $FFMPEG_DIR — only needed for voice cloning (decoding the
# reference audio); basic TTS never touches it. Linux uses the BtbN LGPL build; macOS a
# pinned static arm64 binary (GPL, but we don't redistribute it — your machine fetches it
# from the provider, like the models from HF). Override with ZONOS2_FFMPEG_URL.
ffmpeg_download() {
    mkdir -p "$FFMPEG_DIR" || return 1
    ff_url=""; ff_single=1
    if [ -n "$FFMPEG_URL" ]; then
        ff_url="$FFMPEG_URL"
    elif [ "$(uname -s)" = Darwin ]; then
        ff_url="https://github.com/eugeneware/ffmpeg-static/releases/download/b6.1.1/ffmpeg-darwin-arm64"
    else
        ff_url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-lgpl-7.1.tar.xz"
        ff_single=0
    fi
    echo "downloading ffmpeg ($(ffmpeg_size_mb) MB, for voice cloning) ..."
    if [ "$ff_single" = 1 ]; then
        curl -L --fail --retry 3 --progress-bar -o "$FFMPEG_BIN.part" "$ff_url" || return 1
        chmod +x "$FFMPEG_BIN.part"; mv "$FFMPEG_BIN.part" "$FFMPEG_BIN"
    else
        # BtbN archive nests the binary at ffmpeg-*/bin/ffmpeg; extract, relocate, clean up
        ff_tmp="$FFMPEG_DIR/ffmpeg-dl.tar.xz"; ff_x="$FFMPEG_DIR/.extract"
        rm -rf "$ff_x"; mkdir -p "$ff_x"
        curl -L --fail --retry 3 --progress-bar -o "$ff_tmp" "$ff_url" || { rm -rf "$ff_tmp" "$ff_x"; return 1; }
        tar -xf "$ff_tmp" -C "$ff_x" || { rm -rf "$ff_tmp" "$ff_x"; return 1; }
        ff_found=$(find "$ff_x" -type f -name ffmpeg -path '*/bin/*' 2>/dev/null | head -1)
        [ -n "$ff_found" ] || { rm -rf "$ff_tmp" "$ff_x"; return 1; }
        mv "$ff_found" "$FFMPEG_BIN"; chmod +x "$FFMPEG_BIN"
        rm -rf "$ff_tmp" "$ff_x"
    fi
    [ -x "$FFMPEG_BIN" ]
}

# Sourced by tests/segment-download.sh to exercise download_segmented in isolation; a normal
# run leaves ZONOS2_LIB_ONLY unset and proceeds into the main flow below.
[ "${ZONOS2_LIB_ONLY:-}" = 1 ] && return 0

[ -x "$ZONOS2_SERVER_BIN" ] || die "zonos2-server not found at $ZONOS2_SERVER_BIN (run this script from the extracted release archive, or set ZONOS2_SERVER_BIN)"

BACKBONE="zonos2-$ZONOS2_QUANT.gguf"
MISSING=()
for f in "$BACKBONE" dac.gguf spk-encoder.gguf; do
    [ -f "$ZONOS2_MODEL_DIR/$f" ] || MISSING+=("$f")
done

# ffmpeg is only needed for voice cloning. Prefer our own downloaded copy, then a system
# ffmpeg on PATH; otherwise mark it for download alongside the models.
NEED_FFMPEG=0
if [ -x "$FFMPEG_BIN" ]; then
    export ZONOS2_FFMPEG="$FFMPEG_BIN"
elif [ "${ZONOS2_SKIP_PATH_FFMPEG:-0}" != 1 ] && command -v ffmpeg >/dev/null 2>&1; then
    :   # a system ffmpeg is on PATH; the server/CLI resolve it there (ZONOS2_SKIP_PATH_FFMPEG=1 forces a download, for tests)
else
    NEED_FFMPEG=1
fi

if [ "${#MISSING[@]}" -gt 0 ] || [ "$NEED_FFMPEG" = 1 ]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to download models — install it (e.g. 'sudo apt install curl' / 'brew install curl'), or place the .gguf files in $ZONOS2_MODEL_DIR yourself (see README)"
    echo "The following will be downloaded into $ZONOS2_MODEL_DIR:"
    for f in "${MISSING[@]}"; do
        echo "  $f  ($(model_size_gb "$f") GB)  from $ZONOS2_BASE_URL"
    done
    [ "$NEED_FFMPEG" = 1 ] && echo "  ffmpeg  (~$(ffmpeg_size_mb) MB, voice cloning only)"
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
    if [ "$NEED_FFMPEG" = 1 ]; then
        if ffmpeg_download; then
            export ZONOS2_FFMPEG="$FFMPEG_BIN"
        else
            echo "start-zonos2: warning: ffmpeg download failed — voice cloning disabled (basic TTS still works)" >&2
        fi
    fi
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
