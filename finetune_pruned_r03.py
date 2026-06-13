#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tune the backbone-only r0.3 pruned RTDETRMM checkpoint.

Behavior:
1. Train from the r0.3 pruned .pt checkpoint by default.
2. Disable periodic checkpoints by default (save_period=-1).
3. Keep only best.pt after training.
4. Keep the final checkpoint inside the training run directory only.
5. Save runs under runs_finetune/<model_stem> by default.
6. Use cosine annealing learning rate scheduling.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys
import torch


DEFAULT_MODEL = r"D:\BaiduNetdiskDownload\MutilModel_3398475911\prune_outputs_obc_compare_c3k2\pruned_obc_taylorlike_r0.4_0612_1239.pt"
DEFAULT_DATA = r"D:\BaiduNetdiskDownload\m4FD\M3FD_split\data.yaml"
DEFAULT_PROJECT = Path(__file__).resolve().parent / "runs_finetune"
DEFAULT_EXPECTED_PARAMS = 0


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{ts}] {msg}\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the backbone-only r0.3 pruned RTDETRMM model")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to pruned checkpoint")
    parser.add_argument(
        "--expected-params",
        type=int,
        default=DEFAULT_EXPECTED_PARAMS,
        help="Expected parameter count for the loaded checkpoint. Set <=0 to disable the check.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset yaml path")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Ultralytics project dir")
    parser.add_argument("--name", default="M3F-0.4-dep+obc-150-wu", help="Run name. Empty means using the pruned model filename.")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr0", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--optimizer", default="auto", help="Optimizer name, e.g. auto/AdamW/SGD")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction for quick smoke tests")
    parser.add_argument("--device", default="0", help="Device, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint period, <1 disables epoch checkpoints")
    parser.add_argument(
        "--strict-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop immediately if the loaded checkpoint parameter count does not match --expected-params.",
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Show training logs")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Save training plots")
    parser.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=True, help="Show tqdm progress bar")
    parser.add_argument("--dry-run", action="store_true", help="Only validate paths and model loading")
    return parser.parse_args()


def parse_device(device_str: str):
    ds = str(device_str).strip()
    return int(ds) if ds.isdigit() else ds


def cleanup_and_export_best(project: str, name: str) -> Path:
    run_dir = Path(project) / name
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"

    if not weights_dir.exists():
        raise FileNotFoundError(f"weights directory not found: {weights_dir}")

    for ckpt in weights_dir.glob("epoch*.pt"):
        if ckpt.exists():
            ckpt.unlink()

    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
        last_pt.unlink()

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")
    return best_pt


def resolve_run_name(model_path: Path, name_arg: str) -> str:
    name_arg = str(name_arg).strip()
    if name_arg:
        return name_arg
    return model_path.stem


def inspect_loaded_model(model, expected_params: int, strict_params: bool) -> tuple[int, float]:
    from ultralytics.utils.torch_utils import model_info

    _, n_params, _, flops = model_info(model.model, verbose=True)
    log(f"Loaded checkpoint stats: params={n_params:,}, gflops={flops:.1f}")

    if expected_params > 0 and n_params != expected_params:
        msg = (
            f"Loaded checkpoint parameter count mismatch: got {n_params:,}, "
            f"expected {expected_params:,}. This usually means the script did not load the intended pruned model."
        )
        if strict_params:
            raise RuntimeError(msg)
        log(f"WARNING: {msg}")

    return n_params, flops


def prepare_model_for_finetune(pruned_model: torch.nn.Module) -> None:
    # Some loaded checkpoints carry `criterion=None` and stale frozen flags.
    # Normalize them once before handing the module to Ultralytics' trainer.
    if getattr(pruned_model, "criterion", None) is None and hasattr(pruned_model, "init_criterion"):
        pruned_model.criterion = pruned_model.init_criterion()

    for p in pruned_model.parameters():
        if p.dtype.is_floating_point and not p.requires_grad:
            p.requires_grad = True


def make_pruned_trainer(pruned_model: torch.nn.Module, expected_params: int, strict_params: bool):
    from ultralytics.models.rtdetrmm.train import RTDETRMMTrainer
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import compute_model_gflops, get_num_params, model_info

    class PrunedRTDETRMMTrainer(RTDETRMMTrainer):
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            if isinstance(weights, torch.nn.Module):
                model = weights
            else:
                model = pruned_model

            if model is None:
                raise RuntimeError("Pruned model instance is missing. Cannot continue fine-tuning.")

            prepare_model_for_finetune(model)
            channels = 3 if self.modality else 3 + self.data.get("Xch", 3)
            self.data["channels"] = channels

            try:
                if hasattr(model, "multimodal_router") and model.multimodal_router:
                    model.multimodal_router.update_dataset_config(self.data)
            except Exception as e:
                LOGGER.warning(f"PrunedRTDETRMMTrainer: update_dataset_config failed: {e}")

            n_params = int(get_num_params(model))
            flops = float("nan")
            if verbose:
                info = model_info(model, verbose=True)
                if info is not None:
                    _, _, _, flops = info
            msg = f"Using pre-pruned model instance for fine-tuning: params={n_params:,}"
            if flops == flops:
                msg += f", gflops={flops:.1f}"
            LOGGER.info(msg)

            if expected_params > 0 and n_params != expected_params:
                msg = (
                    f"Trainer received unexpected model structure: got {n_params:,} params, "
                    f"expected {expected_params:,}."
                )
                if strict_params:
                    raise RuntimeError(msg)
                LOGGER.warning(msg)

            try:
                imgsz = int(getattr(self.args, "imgsz", 640))
                arch_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=False)
                if self.modality:
                    route_gflops = compute_model_gflops(model, imgsz=imgsz, modality=self.modality, route_aware=True)
                    LOGGER.info(
                        f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[{self.modality}]): {route_gflops:.2f}"
                    )
                else:
                    route_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=True)
                    LOGGER.info(f"GFLOPs (arch): {arch_gflops:.2f} | GFLOPs (route[dual]): {route_gflops:.2f}")
            except Exception as e:
                LOGGER.warning(f"GFLOPs statistics failed during trainer setup: {e}")

            return model

    return PrunedRTDETRMMTrainer


def self_test_trainer_wiring(
    trainer_cls,
    pruned_model: torch.nn.Module,
    model_path: Path,
    data_path: Path,
    args: argparse.Namespace,
    run_name: str,
) -> tuple[int, float]:
    trainer = trainer_cls(
        overrides={
            "model": str(model_path),
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "optimizer": args.optimizer,
            "cos_lr": True,
            "fraction": args.fraction,
            "device": parse_device(args.device),
            "workers": min(args.workers, 2),
            "project": str(args.project),
            "name": f"{run_name}_dryrun",
            "save_period": -1,
            "plots": False,
            "verbose": False,
            "exist_ok": True,
        }
    )
    wired_model = trainer.get_model(cfg=getattr(pruned_model, "yaml", None), weights=pruned_model, verbose=False)
    if wired_model is not pruned_model:
        raise RuntimeError("Trainer wiring test failed: training did not reuse the loaded pruned model instance.")

    n_params = sum(p.numel() for p in wired_model.parameters())
    try:
        from ultralytics.utils.torch_utils import compute_model_gflops

        flops = float(compute_model_gflops(wired_model, imgsz=args.imgsz, modality=None, route_aware=False))
    except Exception:
        flops = float("nan")
    return n_params, flops


def main() -> int:
    args = parse_args()
    os.environ["TQDM_DISABLE"] = "False" if args.tqdm else "True"
    os.environ["YOLO_VERBOSE"] = "True" if args.verbose else "False"
    os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    if not str(args.model).strip():
        raise FileNotFoundError("请通过 --model 指定剪枝模型，或先在脚本里的 DEFAULT_MODEL 填好路径。")

    model_path = Path(args.model).expanduser()
    data_path = Path(args.data)
    run_name = resolve_run_name(model_path, args.name)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"data yaml not found: {data_path}")

    from ultralytics import RTDETRMM

    log(f"Loading pruned model: {model_path}")
    model = RTDETRMM(str(model_path))
    inspect_loaded_model(model, args.expected_params, args.strict_params)
    prepare_model_for_finetune(model.model)
    trainer_cls = make_pruned_trainer(model.model, args.expected_params, args.strict_params)
    wired_params, wired_flops = self_test_trainer_wiring(
        trainer_cls, model.model, model_path, data_path, args, run_name
    )
    if wired_flops == wired_flops:
        log(f"Trainer wiring check passed: params={wired_params:,}, gflops={wired_flops:.1f}")
    else:
        log(f"Trainer wiring check passed: params={wired_params:,}, gflops=nan")

    if args.dry_run:
        log("Dry run passed: model/data path, checkpoint loading, and pruned-trainer wiring are OK.")
        return 0

    log("Training started.")
    model.train(
        trainer=trainer_cls,
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        optimizer=args.optimizer,
        cos_lr=True,
        fraction=args.fraction,
        device=parse_device(args.device),
        workers=args.workers,
        project=args.project,
        name=run_name,
        exist_ok=True,
        verbose=args.verbose,
        plots=args.plots,
        save_period=args.save_period,
    )

    best_path = cleanup_and_export_best(args.project, run_name)
    log(f"Training finished. Best model saved to: {best_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user.")
        raise SystemExit(130)
