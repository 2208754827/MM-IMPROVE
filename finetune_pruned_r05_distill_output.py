#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


DEFAULT_MODEL = Path(r"D:\BaiduNetdiskDownload\MutilModel_3398475911\runs_finetune\M3F-0.5-dep+ty-150-wu\weights\best.pt")
DEFAULT_DATA = Path(r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml")
DEFAULT_DISTILL = Path(r"D:\BaiduNetdiskDownload\MutilModel_3398475911\distill_r05_output.yaml")
DEFAULT_PROJECT = Path(r"D:\JiQI\MM-experiment\DistillRT")
DEFAULT_NAME = "r05-outputdistill-cmxfusion2-teacher"
DEFAULT_EXPECTED_PARAMS = 7966554


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    sys.stdout.write(f"[{ts}] {msg}\n")
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune r0.5 pruned RTDETRMM with output distillation.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Pruned student checkpoint")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset yaml path")
    parser.add_argument("--distill-yaml", default=str(DEFAULT_DISTILL), help="Distillation yaml path")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT), help="Output project dir")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Run name")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr0", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--optimizer", default="auto", help="Optimizer name")
    parser.add_argument("--device", default="0", help="Device id or cpu")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction")
    parser.add_argument("--mode", default="output", choices=["output", "feature", "both"], help="Distill mode")
    parser.add_argument("--save-period", type=int, default=-1, help="Checkpoint save period")
    parser.add_argument("--expected-params", type=int, default=DEFAULT_EXPECTED_PARAMS, help="Expected student params")
    parser.add_argument("--dry-run", action="store_true", help="Only validate paths, teacher config and trainer wiring")
    return parser.parse_args()


def prepare_env() -> None:
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")
    os.environ.setdefault("YOLO_MM_QUIET_PROGRESS", "true")


def prepare_model_for_finetune(pruned_model: torch.nn.Module) -> None:
    if getattr(pruned_model, "criterion", None) is None and hasattr(pruned_model, "init_criterion"):
        pruned_model.criterion = pruned_model.init_criterion()
    for p in pruned_model.parameters():
        if p.dtype.is_floating_point and not p.requires_grad:
            p.requires_grad = True


def parse_device(device_str: str):
    ds = str(device_str).strip()
    return int(ds) if ds.isdigit() else ds


def load_and_validate_student(model_path: Path, expected_params: int) -> torch.nn.Module:
    from ultralytics import RTDETRMM
    from ultralytics.nn.tasks import RTDETRDetectionModel

    student_wrapper = RTDETRMM(str(model_path))
    student_model = student_wrapper.model.float().eval()
    if not isinstance(student_model, RTDETRDetectionModel):
        raise TypeError(f"Student is not RTDETRDetectionModel: {type(student_model).__name__}")

    params = sum(p.numel() for p in student_model.parameters())
    if expected_params > 0 and params != expected_params:
        raise RuntimeError(f"Unexpected student params: got {params:,}, expected {expected_params:,}")

    log(f"Validated student: {model_path}")
    log(f"Student params: {params:,}")
    return student_model


def validate_teacher_and_distill(distill_path: Path) -> None:
    from ultralytics.nn.mm.distill.schema import load_distill_config
    from ultralytics.nn.tasks import attempt_load_one_weight, RTDETRDetectionModel

    cfg = load_distill_config(str(distill_path))
    if not cfg.teachers:
        raise RuntimeError("Distill config contains no teachers.")

    teacher_spec = cfg.teachers[0]
    teacher_weights = Path(teacher_spec.weights)
    if not teacher_weights.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_weights}")

    teacher_model, _ = attempt_load_one_weight(str(teacher_weights))
    if not isinstance(teacher_model, RTDETRDetectionModel):
        raise TypeError(f"Teacher is not RTDETRDetectionModel: {type(teacher_model).__name__}")

    log(f"Validated teacher: {teacher_weights}")


def make_pruned_distill_trainer(pruned_model: torch.nn.Module, expected_params: int):
    from ultralytics.models.rtdetrmm.train import RTDETRMMTrainer
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import compute_model_gflops, get_num_params, model_info

    class PrunedDistillRTDETRMMTrainer(RTDETRMMTrainer):
        def get_model(self, cfg=None, weights=None, verbose: bool = True):
            model = pruned_model
            if model is None:
                raise RuntimeError("Pruned model instance is missing.")

            prepare_model_for_finetune(model)
            channels = 3 if self.modality else 3 + self.data.get("Xch", 3)
            self.data["channels"] = channels

            try:
                if hasattr(model, "multimodal_router") and model.multimodal_router:
                    model.multimodal_router.update_dataset_config(self.data)
            except Exception as e:
                LOGGER.warning(f"PrunedDistillRTDETRMMTrainer: update_dataset_config failed: {e}")

            n_params = int(get_num_params(model))
            flops = float("nan")
            if verbose:
                info = model_info(model, verbose=True)
                if info is not None:
                    _, _, _, flops = info

            LOGGER.info(f"Using pre-pruned student model for distillation: params={n_params:,}")
            if expected_params > 0 and n_params != expected_params:
                raise RuntimeError(f"Trainer received unexpected student params: got {n_params:,}, expected {expected_params:,}")

            try:
                imgsz = int(getattr(self.args, "imgsz", 640))
                arch_gflops = compute_model_gflops(model, imgsz=imgsz, modality=None, route_aware=False)
                LOGGER.info(f"Student GFLOPs (arch): {arch_gflops:.2f}")
            except Exception as e:
                LOGGER.warning(f"GFLOPs statistics failed during trainer setup: {e}")

            return model

    return PrunedDistillRTDETRMMTrainer


def self_test_trainer_wiring(trainer_cls, pruned_model: torch.nn.Module, model_path: Path, data_path: Path, args) -> None:
    dryrun_project = Path(__file__).resolve().parent / "_distill_dryrun"
    trainer = trainer_cls(
        overrides={
            "model": str(model_path),
            "data": str(data_path),
            "epochs": 1,
            "imgsz": args.imgsz,
            "batch": 1,
            "lr0": args.lr0,
            "optimizer": args.optimizer,
            "cos_lr": True,
            "fraction": min(args.fraction, 0.001),
            "device": parse_device(args.device),
            "workers": 0,
            "project": str(dryrun_project),
            "name": f"{args.name}_dryrun",
            "save_period": -1,
            "plots": False,
            "verbose": False,
            "exist_ok": True,
            "distill": [str(args.distill_yaml), args.mode],
        }
    )
    wired_model = trainer.get_model(cfg=getattr(pruned_model, "yaml", None), weights=pruned_model, verbose=False)
    if wired_model is not pruned_model:
        raise RuntimeError("Trainer wiring test failed: distill training did not reuse the pruned student model.")


def main() -> int:
    args = parse_args()
    prepare_env()

    model_path = Path(args.model)
    data_path = Path(args.data)
    distill_path = Path(args.distill_yaml)

    if not model_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")
    if not distill_path.exists():
        raise FileNotFoundError(f"Distillation yaml not found: {distill_path}")

    validate_teacher_and_distill(distill_path)
    student_model = load_and_validate_student(model_path, args.expected_params)
    trainer_cls = make_pruned_distill_trainer(student_model, args.expected_params)
    self_test_trainer_wiring(trainer_cls, student_model, model_path, data_path, args)

    if args.dry_run:
        log("Dry-run validation passed.")
        return 0

    from ultralytics import RTDETRMM

    log(f"Start distill training: student={model_path.name}, mode={args.mode}, run={args.name}")
    wrapper = RTDETRMM(str(model_path))
    wrapper.model = student_model
    wrapper.train(
        trainer=trainer_cls,
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        optimizer=args.optimizer,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        save_period=args.save_period,
        fraction=args.fraction,
        deterministic=False,
        amp=True,
        cos_lr=True,
        distill=[str(distill_path), args.mode],
    )

    weights_dir = Path(args.project) / args.name / "weights"
    log(f"Training finished. Weights dir: {weights_dir}")
    log(f"Best checkpoint: {weights_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
