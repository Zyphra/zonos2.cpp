#!/usr/bin/env bash
# Build a multi-speaker, multi-path KLD reference corpus + manifest for zonos2-perplexity.
#
# Two stages:
#   1. Synthesize K distinct voices: free-run the f16 model (no speaker, varied seeds) to a wav,
#      then encode each wav back to a speaker embedding via spk-encoder-cli --clone. These are
#      model-derived (not real) speakers -- fine for relative quant comparison.
#   2. Generate teacher-forcing id traces (--dump-ids) across a SAMPLED design of
#      speaker x conditioning-path x text, and emit out/kld.manifest (one line per trace:
#      `ids.npy [speaker.npy [spk_pos]]`). No-speaker lines exercise the plain-TTS path; cloned
#      lines (spk_pos=0) exercise spk_lda/spk_proj and the speaker-conditioned routes.
#
# Conditioning paths covered: default, speaking-rate (slow/fast), quality-bucket override
# (no-speaker); default, inaccurate, noisy-bg, fast-rate (cloned -- where bg/accurate rows exist).
# The design is sampled, not a full grid; coverage is printed at the end (no silent truncation).
#
# Then build the base:
#   zonos2-perplexity out/zonos2-f16.gguf --kl-divergence-base out/ref-multi.kld.bin \
#       --manifest out/kld.manifest --gpu
set -euo pipefail
cd /data/home/sofian/zonos2.cpp

CLI=./build-cuda/zonos2-cli
SPKCLI=./build-cuda/spk-encoder-cli
MODEL=out/zonos2-f16.gguf
DAC=out/dac.gguf
SPKGGUF=out/spk-encoder.gguf
OUT=out/kldcorp
MAN=out/kld.manifest
GPU=${CUDA_VISIBLE_DEVICES:-0}
NVOICES=${NVOICES:-4}
TRACE_MAX=${TRACE_MAX:-300}
VOICE_MAX=${VOICE_MAX:-200}

mkdir -p "$OUT"
rm -f "$OUT"/* "$MAN" 2>/dev/null || true
: > "$MAN"

texts=(
"The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains."
"Why do we dream? Neuroscientists have debated this question for over a century without consensus."
"In nineteen eighty-five, researchers discovered a new species of deep-sea creature near the vents."
"Please remember to bring your umbrella, your charger, your passport, and the blue folder tomorrow."
"What a beautiful morning! The birds are singing and the air smells fresh after last night's storm."
"Our quarterly revenue grew by twelve percent, driven largely by strong international demand."
"She whispered softly, but her words carried across the empty concert hall like a haunting melody."
"Turn left at the second traffic light, continue for half a mile, and the museum is on your right."
)
ntext=${#texts[@]}

run() { CUDA_VISIBLE_DEVICES=$GPU "$CLI" "$MODEL" "$@" --gpu >/dev/null 2>&1; }

# ---- stage 1: synthesize voices -> speaker embeddings ----
echo "=== stage 1: synthesizing $NVOICES voices (GPU $GPU) ==="
spk_npys=()
for v in $(seq 0 $((NVOICES-1))); do
  seed=$((7 + v*10))
  vbase="$OUT/voice$v"
  run --tts "${texts[$((v % ntext))]}" "$vbase.npy" --dac "$DAC" --seed "$seed" --max "$VOICE_MAX" \
    && CUDA_VISIBLE_DEVICES=$GPU "$SPKCLI" "$SPKGGUF" --clone "$vbase.npy.wav" "$OUT/spk_v$v.npy" >/dev/null 2>&1 \
    && { spk_npys+=("$OUT/spk_v$v.npy"); echo "  voice$v ok (seed $seed)"; } \
    || echo "  voice$v FAILED"
done
nspk=${#spk_npys[@]}
[ "$nspk" -gt 0 ] || { echo "no voices synthesized; aborting"; exit 1; }

# ---- stage 2: generate traces + manifest ----
idx=0
n_nospk=0; n_spk=0
emit() {  # emit <text> <seed> <manifest-line-suffix> [cli cond flags...]
  local text="$1" seed="$2" suffix="$3"; shift 3
  local ids="$OUT/seq_$(printf '%03d' "$idx").npy"
  local codes="$OUT/seq_$(printf '%03d' "$idx").codes.npy"
  if run --tts "$text" "$codes" --dump-ids "$ids" --seed "$seed" --max "$TRACE_MAX" "$@"; then
    echo "$ids$suffix" >> "$MAN"
    idx=$((idx+1)); return 0
  fi
  echo "  trace $idx FAILED ($suffix)"; return 1
}

echo "=== stage 2a: no-speaker conditioning paths ==="
# variants applied to plain TTS (rate + quality affect prompt without a speaker slot)
nospk_variants=( "default::" "rate-slow:--speaking-rate 1:" "rate-fast:--speaking-rate 6:" "quality-alt:--quality 0:0:" )
ti=0
for var in "${nospk_variants[@]}"; do
  name="${var%%:*}"; flags="${var#*:}"; flags="${flags%:}"
  for rep in 0 1; do
    t=$(( ti % ntext )); ti=$((ti+1)); seed=$((1000 + idx))
    # shellcheck disable=SC2086
    emit "${texts[$t]}" "$seed" "" $flags && { n_nospk=$((n_nospk+1)); echo "  [nospk/$name] text$t seed$seed"; }
  done
done

echo "=== stage 2b: cloned-speaker conditioning paths ==="
# variants meaningful with a speaker slot present (bg/accurate rows live in the speaker block)
spk_variants=( "default:" "inaccurate:--inaccurate" "noisy-bg:--noisy-bg" "fast:--speaking-rate 6" )
vi=0
for sp in "${spk_npys[@]}"; do
  for var in "${spk_variants[@]}"; do
    name="${var%%:*}"; flags="${var#*:}"
    t=$(( vi % ntext )); vi=$((vi+1)); seed=$((2000 + idx))
    # shellcheck disable=SC2086
    emit "${texts[$t]}" "$seed" " $sp 0" --speaker "$sp" $flags \
      && { n_spk=$((n_spk+1)); echo "  [spk/$name] $(basename "$sp") text$t seed$seed"; }
  done
done

echo "=== corpus ready ==="
echo "voices synthesized : $nspk / $NVOICES"
echo "no-speaker traces  : $n_nospk  (paths: default, rate-slow, rate-fast, quality-alt)"
echo "cloned traces      : $n_spk   (paths: default, inaccurate, noisy-bg, fast x $nspk speakers)"
echo "total sequences    : $idx  -> $MAN"
echo
echo "next: CUDA_VISIBLE_DEVICES=$GPU zonos2-perplexity $MODEL --kl-divergence-base out/ref-multi.kld.bin --manifest $MAN --gpu"
