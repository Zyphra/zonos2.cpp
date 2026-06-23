# ggml patches

Local fixes carried on top of the upstream `ggml` submodule **without forking or
vendoring it**. The submodule stays pinned to a pristine upstream SHA; these
patches are (re)applied to its working tree at CMake **configure** time
(see the `find_package(Git)` / `git apply` block in the top-level `CMakeLists.txt`).

Each `*.patch` is a `git diff` taken from the **ggml repo root** (paths look like
`a/src/ggml-cuda/...`). Application is idempotent: configure first runs
`git apply --reverse --check`; if that succeeds the tree is already patched and
the step is skipped, so re-running cmake never double-applies. Patches are applied
in sorted filename order.

## Current patches

- `conv-transpose-1d-fix.patch` — the CUDA `conv_transpose_1d` kernel scanned the
  entire input length per output element (O(L²)); for DAC decode of long
  sequences this dominated runtime (>470s for a 2048-frame clip). The fix computes
  the contributing tap range directly (O(L·K)). Bit-identical output, ~240× faster.
  Worth upstreaming.
- `metal-ncb-env.patch` — let the Metal backend read the command-buffer count from
  `GGML_METAL_NCB` (upstream hard-codes 1). Single-token decode is a chain of ~1675
  tiny kernels; with one command buffer the ~3 ms CPU encode runs nearly serially
  ahead of the GPU, but spreading it over a few buffers lets the GPU stream through
  them as they enqueue — ~10% faster decode at no quality cost. `zonos2_model_load`
  defaults it to 4 for the GPU backbone (overridable); prefill's big GEMMs are
  unaffected. Behavior is identical to upstream when the env var is unset.

## Caveats

- Patches are applied at **configure** time. If you reset the submodule worktree
  (`git submodule update`) without re-running cmake, an incremental build would use
  unpatched sources — re-run `cmake` (configure) after any submodule reset.
- After a submodule **bump**, a patch may no longer apply against the new sources;
  configure then fails with a `FATAL_ERROR`. Refresh the patch: revert the worktree
  (`git -C ggml checkout <file>`), re-apply the change by hand against the new code,
  and regenerate with `git -C ggml diff <file> > patches/<name>.patch`.
