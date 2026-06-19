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

## Caveats

- Patches are applied at **configure** time. If you reset the submodule worktree
  (`git submodule update`) without re-running cmake, an incremental build would use
  unpatched sources — re-run `cmake` (configure) after any submodule reset.
- After a submodule **bump**, a patch may no longer apply against the new sources;
  configure then fails with a `FATAL_ERROR`. Refresh the patch: revert the worktree
  (`git -C ggml checkout <file>`), re-apply the change by hand against the new code,
  and regenerate with `git -C ggml diff <file> > patches/<name>.patch`.
