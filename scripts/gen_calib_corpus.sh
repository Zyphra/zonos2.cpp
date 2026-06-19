#!/usr/bin/env bash
# Build a generation-trace imatrix calibration corpus: free-run the f16 model on diverse
# prompts (sampled, varied seeds) and dump the full teacher-forcing ids it consumed. Unlike
# text-prompt prefill, these traces exercise the audio-generation experts that real inference
# uses. A subset is regenerated with a cloned speaker so cloning routes are covered too.
set -euo pipefail
cd /data/home/sofian/zonos2.cpp

CLI=./build-cuda/zonos2-cli
MODEL=out/zonos2-f16.gguf
SPK=out/golden_spk/speaker.npy
OUT=out/gencal
GPU=${CUDA_VISIBLE_DEVICES:-0}
MAX=${MAX:-300}
mkdir -p "$OUT"
rm -f "$OUT"/seq*.npy "$OUT"/seq*.eos.npy 2>/dev/null || true

texts=(
"The quick brown fox jumps over the lazy dog while the sun sets behind the distant mountains."
"In nineteen eighty-five, researchers discovered a new species of deep-sea creature near the volcanic vents."
"She whispered softly, but her words carried across the empty concert hall like a haunting melody."
"Economic forecasts suggest that inflation may continue to rise throughout the coming fiscal quarter."
"Once upon a time, in a kingdom far beyond the northern seas, there lived a curious young inventor."
"Please remember to bring your umbrella, your charger, your passport, and the blue folder tomorrow."
"The chemical reaction produced an unexpected burst of violet light and a faint smell of citrus."
"Why do we dream? Neuroscientists have debated this question for over a century without consensus."
"Breaking news: the championship match has been postponed due to severe weather across the region."
"He counted slowly from one to twenty, then opened his eyes and began to search the quiet garden."
"Wait! Don't touch that wire, it's still connected to the main power supply and could be dangerous."
"I absolutely loved the film, though the ending left me with so many unanswered questions."
"The recipe calls for two cups of flour, three eggs, a pinch of salt, and a teaspoon of vanilla."
"Ladies and gentlemen, thank you for joining us this evening for a very special performance."
"Honestly, I can't believe it's already June; this year has flown by faster than I expected."
"The spacecraft entered orbit at exactly four thirty in the morning, transmitting data back home."
"Could you please repeat the address? I want to make absolutely sure I wrote it down correctly."
"Thunder rolled across the valley as the first heavy drops of rain began to strike the rooftops."
"Our quarterly revenue grew by twelve percent, driven largely by strong international demand."
"What a beautiful morning! The birds are singing and the air smells fresh after last night's storm."
"He paused at the doorway, took a deep breath, and finally stepped into the crowded interview room."
"The ancient manuscript described a hidden chamber beneath the temple, guarded by a stone serpent."
"Turn left at the second traffic light, continue for half a mile, and the museum will be on your right."
"I'm so sorry for the delay; the train was held up because of a signal failure outside the station."
"Mathematics is the language in which the universe writes its most elegant and enduring secrets."
"The toddler giggled with delight as the puppy chased its own tail around and around the living room."
"According to the report, global temperatures have risen by roughly one degree over the last century."
"Grab your coat, lock the door, and meet me by the old oak tree at the bottom of the hill at noon."
"The orchestra tuned their instruments while the audience settled into their plush velvet seats."
"Remember, courage is not the absence of fear, but the decision that something else matters more."
)

n=${#texts[@]}
echo "=== plain TTS traces: $n prompts, max $MAX frames each (GPU $GPU) ==="
for i in "${!texts[@]}"; do
  seed=$((i + 1))
  CUDA_VISIBLE_DEVICES=$GPU "$CLI" "$MODEL" --tts "${texts[$i]}" "$OUT/seq_t$(printf '%02d' "$i").codes.npy" \
    --dump-ids "$OUT/seq_t$(printf '%02d' "$i").npy" --max "$MAX" --seed "$seed" --gpu \
    >/dev/null 2>&1 && echo "  t$i done (seed $seed)" || echo "  t$i FAILED"
done

# Speaker-cloning subset: first 12 prompts, regenerated with a cloned voice + fresh seeds so
# the cloning-conditioned routes (which differ from plain TTS) also land in the corpus.
echo "=== speaker-cloning traces: 12 prompts ==="
for i in $(seq 0 11); do
  seed=$((100 + i))
  CUDA_VISIBLE_DEVICES=$GPU "$CLI" "$MODEL" --tts "${texts[$i]}" "$OUT/seq_s$(printf '%02d' "$i").codes.npy" \
    --dump-ids "$OUT/seq_s$(printf '%02d' "$i").npy" --speaker "$SPK" --max "$MAX" --seed "$seed" --gpu \
    >/dev/null 2>&1 && echo "  s$i done (seed $seed)" || echo "  s$i FAILED"
done

echo "=== corpus ready ==="
ls "$OUT"/seq_*.npy | grep -v codes | wc -l
echo "total rows:"; /data/home/sofian/ZONOS2/.venv/bin/python - <<'PY'
import glob, numpy as np
tot=0
for f in sorted(glob.glob('out/gencal/seq_*.npy')):
    if 'codes' in f: continue
    tot += np.load(f).shape[0]
print(tot)
PY
