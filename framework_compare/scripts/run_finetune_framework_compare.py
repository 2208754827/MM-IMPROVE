#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune a pruned framework-compare checkpoint into this folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compare_common import DEFAULT_DATA, ROOT, build_stage_project_dir, default_run_name, resolve_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pruned checkpoint for framework comparison.")
    parser.add_argument("--model", required=True, help="Path to pruned checkpoint")
    parser.add_argument("--name", default="", help="Optional run name; defaults to the pruned checkpoint stem")
    parser.add_argument("--framework", default="", help="Optional framework tag override, e.g. depgraph/bn_slimming/random")
    parser.add_argument("--prune-ratio", type=float, default=None, help="Optional prune ratio override, e.g. 0.4")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=5e-5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Pruned checkpoint not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data yaml not found: {data_path}")

    framework, ratio_tag = resolve_context(model_path, args.framework, args.prune_ratio)
    project_dir = build_stage_project_dir("finetune", framework, ratio_tag)
    run_name = default_run_name(model_path, args.name)

    sys.path.insert(0, str(ROOT))
    import finetune_pruned_r03

    sys.argv = [
        "finetune_pruned_r03.py",
        "--model", str(model_path),
        "--data", str(data_path),
        "--project", str(project_dir),
        "--name", run_name,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--imgsz", str(args.imgsz),
        "--lr0", str(args.lr0),
        "--device", args.device,
        "--workers", str(args.workers),
        "--optimizer", args.optimizer,
        "--fraction", str(args.fraction),
    ]
    return finetune_pruned_r03.main()


if __name__ == "__main__":
    raise SystemExit(main())
