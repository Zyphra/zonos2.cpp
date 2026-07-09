#!/usr/bin/env python3
"""Normalize TTS text through the reference ZONOS2 NeMo text normalizer.

This is intentionally a tiny helper for zonos2-server. The C++ server invokes it
only when text normalization is explicitly enabled with --text-normalizer-python.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import pathlib
import sys
import types


def _find_zonos2_python(explicit: str | None) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit).expanduser())
    repo = pathlib.Path(__file__).resolve().parents[1]
    candidates.append(repo.parent / "ZONOS2" / "python")
    for path in candidates:
        if (path / "zonos2" / "tokenizer" / "textnorm.py").exists():
            return path
    raise RuntimeError(
        "could not find ZONOS2/python; pass --zonos2-python or set up a sibling ZONOS2 checkout"
    )


def _load_textnorm(zonos2_python: pathlib.Path):
    # textnorm.py only needs init_logger from zonos2.utils, but importing the
    # package utility module pulls in multiprocessing/zmq/torch-adjacent deps.
    utils = types.ModuleType("zonos2.utils")
    utils.init_logger = lambda name: logging.getLogger(name)
    sys.modules.setdefault("zonos2.utils", utils)

    sys.path.insert(0, str(zonos2_python))
    path = zonos2_python / "zonos2" / "tokenizer" / "textnorm.py"
    spec = importlib.util.spec_from_file_location("zonos2_textnorm_direct", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en_us")
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--zonos2-python", default=None)
    args = parser.parse_args()

    text = pathlib.Path(args.input).read_text(encoding="utf-8")
    zonos2_python = _find_zonos2_python(args.zonos2_python)
    mod = _load_textnorm(zonos2_python)
    normalizer = mod.TTSTextNormalizer(args.cache_dir)
    sys.stdout.write(normalizer.normalize(text, args.language))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
