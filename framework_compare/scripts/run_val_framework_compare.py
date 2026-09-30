#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate a framework-compare checkpoint into this folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compare_common import DEFAULT_DATA, EXP_ROOT, ROOT, default_run_name, build_stage_project_dir, resolve_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate checkpoint for framework comparison.")
    parser.add_argument("--model", required=True, help="Path to checkpoint")
    parser.add_argument("--name", default="", help="Optional validation run name; defaults to the checkpoint stem")
    parser.add_argument("--framework", default="", help="Optional framework tag override, e.g. depgraph/bn_slimming/random")
    parser.add_argument("--prune-ratio", type=float, default=None, help="Optional prune ratio override, e.g. 0.4")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_path}")

    framework, ratio_tag = resolve_context(model_path, args.framework, args.prune_ratio)
    project_dir = build_stage_project_dir("val", framework, ratio_tag)
    run_name = default_run_name(model_path, args.name)

    sys.path.insert(0, str(ROOT))
    from ultralytics import RTDETRMM

    model = RTDETRMM(str(model_path))
    model.val(
        data=str(data_path),
        split=args.split,
        device=args.device,
        project=str(project_dir),
        name=run_name,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
