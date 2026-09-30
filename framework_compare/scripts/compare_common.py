#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared helpers for standardized framework comparison experiments."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = EXP_ROOT / "outputs"
DEFAULT_WEIGHTS = Path(r"D:\JiQI\MM-experiment\ResTest\A-DWConv-C3k2_PConv\weights\best.pt")
DEFAULT_DATA = Path(r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml")

_FRAMEWORK_ALIASES = {
    "dep": "depgraph",
    "depgraph": "depgraph",
    "taylor": "depgraph",
    "unstructured": "unstructured",
    "magnitude": "unstructured",
    "random": "random",
    "bnslim_train": "bnslim_train",
    "bn_slimming_train": "bnslim_train",
    "bnslim": "bnslim_train",
    "bn": "bnslim_train",
}


def normalize_framework(name: str | None) -> str:
    if not name:
        return "unknown"
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return _FRAMEWORK_ALIASES.get(key, key)


def ratio_tag(ratio: float) -> str:
    return f"r{int(round(float(ratio) * 100)):03d}"


def build_prune_output_dir(framework: str, ratio: float) -> Path:
    return OUTPUT_ROOT / "prune" / normalize_framework(framework) / ratio_tag(ratio)


def build_stage_project_dir(stage: str, framework: str, ratio_value: float | str) -> Path:
    if isinstance(ratio_value, str):
        rtag = ratio_value if ratio_value.startswith("r") else ratio_tag(float(ratio_value))
    else:
        rtag = ratio_tag(ratio_value)
    return OUTPUT_ROOT / stage / normalize_framework(framework) / rtag


def _extract_ratio_from_text(text: str) -> float | None:
    s = str(text).lower()
    m = re.search(r"\br(\d{3})\b", s)
    if m:
        return int(m.group(1)) / 100.0
    m = re.search(r"(?:^|[_-])r(\d{2})(?:[_-]|$)", s)
    if m:
        return int(m.group(1)) / 100.0
    m = re.search(r"(?:^|[_-])r(0\.\d+)(?:[_-]|$)", s)
    if m:
        return float(m.group(1))
    return None


def infer_framework_ratio_from_model_path(model_path: Path) -> tuple[str, str]:
    framework = "unknown"
    ratio = None

    for part in model_path.parts:
        normalized = normalize_framework(part)
        if normalized in {"depgraph", "unstructured", "random", "bnslim_train"}:
            framework = normalized
        if ratio is None:
            ratio = _extract_ratio_from_text(part)

    if ratio is None:
        ratio = _extract_ratio_from_text(model_path.stem)
    if ratio is None:
        raise ValueError(f"Unable to infer prune ratio from model path: {model_path}")
    return framework, ratio_tag(ratio)


def resolve_context(model_path: Path, framework_arg: str | None, ratio_arg: float | None) -> tuple[str, str]:
    inferred_framework, inferred_ratio_tag = infer_framework_ratio_from_model_path(model_path)
    framework = normalize_framework(framework_arg) if framework_arg else inferred_framework
    ratio = ratio_tag(ratio_arg) if ratio_arg is not None else inferred_ratio_tag
    return framework, ratio


def default_run_name(model_path: Path, name_arg: str | None) -> str:
    value = (name_arg or "").strip()
    return value if value else model_path.stem
