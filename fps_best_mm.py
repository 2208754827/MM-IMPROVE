import argparse
import time
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL_NAME = "aifi-dattention-CSP-MutilScaleEdgeInformationEnhance-ASF-P2-lite"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark multimodal best.pt speed (paper-style infer-only by default)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to best.pt. If empty, script resolves from --model-name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name under ResTest/<name>/weights/best.pt, or a direct .pt path",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--device", type=str, default="0", help="Device, e.g. 0/cuda:0/cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for paper-style infer-only benchmark")
    parser.add_argument("--half", action="store_true", help="Use fp16 (CUDA only)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--max-samples", type=int, default=300, help="Benchmark steps")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (for e2e mode)")
    parser.add_argument(
        "--mode",
        type=str,
        default="paper",
        choices=["paper", "e2e", "both"],
        help="paper=infer-only, e2e=full predict, both=report both",
    )
    parser.add_argument(
        "--save-txt",
        type=str,
        default="",
        help="Append compact results to a text file (optional).",
    )
    return parser.parse_args()


def _resolve_split_dir(root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return (root / split_path).resolve()


def _detect_x_modality(data_cfg: dict) -> str:
    models = data_cfg.get("models", [])
    if isinstance(models, list):
        for m in models:
            if str(m).lower() != "rgb":
                return str(m).lower()

    modalities = data_cfg.get("modalities") or data_cfg.get("modality") or {}
    if isinstance(modalities, dict):
        for k in modalities.keys():
            if str(k).lower() != "rgb":
                return str(k).lower()

    return "ir"


def _build_x_split_dir(root: Path, data_cfg: dict, split: str, rgb_split_dir: Path) -> Path:
    modalities = data_cfg.get("modalities") or data_cfg.get("modality") or {}
    rgb_key = None
    for k in modalities.keys():
        if str(k).lower() == "rgb":
            rgb_key = k
            break

    x_mod = _detect_x_modality(data_cfg)
    x_key = None
    for k in modalities.keys():
        if str(k).lower() == x_mod:
            x_key = k
            break

    if x_key is None:
        return (root / f"images_{x_mod}" / split).resolve()

    x_base = Path(str(modalities[x_key]))
    x_base_abs = x_base if x_base.is_absolute() else (root / x_base).resolve()

    if rgb_key is not None:
        rgb_base = Path(str(modalities[rgb_key]))
        rgb_base_abs = rgb_base if rgb_base.is_absolute() else (root / rgb_base).resolve()
        try:
            suffix = rgb_split_dir.resolve().relative_to(rgb_base_abs)
            return (x_base_abs / suffix).resolve()
        except Exception:
            pass

    return (x_base_abs / split).resolve()


def collect_pairs(data_yaml: Path, split: str):
    with open(data_yaml, "r", encoding="utf-8", errors="ignore") as f:
        data_cfg = yaml.safe_load(f)

    root = Path(data_cfg.get("path", data_yaml.parent)).resolve()
    split_value = data_cfg.get(split, None)
    if split_value is None:
        raise ValueError(f"split='{split}' not found in {data_yaml}")

    rgb_dir = _resolve_split_dir(root, str(split_value))
    x_dir = _build_x_split_dir(root, data_cfg, split, rgb_dir)

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB split dir not found: {rgb_dir}")
    if not x_dir.exists():
        raise FileNotFoundError(f"X split dir not found: {x_dir}")

    rgb_files = sorted([p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    x_files = [p for p in x_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    x_by_name = {p.name: p for p in x_files}
    x_by_stem = {p.stem: p for p in x_files}

    pairs = []
    for rgb in rgb_files:
        x = x_by_name.get(rgb.name, None) or x_by_stem.get(rgb.stem, None)
        if x is not None:
            pairs.append((rgb, x))
    return pairs, rgb_dir, x_dir


def _resolve_device(device_arg, torch):
    d = str(device_arg).strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d.isdigit():
        return torch.device(f"cuda:{d}")
    if d.startswith("cuda"):
        return torch.device(d)
    return torch.device(d)


def _resolve_stride(model, torch):
    stride = getattr(model, "stride", 32)
    if isinstance(stride, int):
        return max(1, int(stride))
    if isinstance(stride, (list, tuple)):
        return max(1, int(max(stride)))
    if torch.is_tensor(stride):
        return max(1, int(stride.max().item()))
    return 32


def build_batches_for_paper(model, pairs, imgsz, batch, warmup, steps, device, half, x_modality_hint="ir"):
    import torch
    from ultralytics.data.multimodal import MultiModalInferenceDataset

    xch = int(getattr(model.mm_router, "INPUT_SOURCES", {}).get("X", 3))
    x_modality = str(getattr(model.mm_router, "x_modality_type", "ir"))
    if x_modality.lower() in {"unknown", "none", ""}:
        x_modality = str(x_modality_hint)
    stride = _resolve_stride(model, torch)

    need_steps = max(1, warmup + steps)
    need_images = min(len(pairs), max(batch, need_steps * batch))
    use_pairs = pairs[:need_images]
    sample_specs = [{"id": i, "rgb_path": rgb, "x_path": x} for i, (rgb, x) in enumerate(use_pairs)]

    dataset = MultiModalInferenceDataset(
        samples=sample_specs,
        imgsz=imgsz,
        dataset_config={"Xch": xch, "x_modality": x_modality},
        stride=stride,
        verbose=False,
    )

    cpu_samples = []
    for i in range(len(dataset)):
        im = dataset[i]["im"].squeeze(0).contiguous()
        cpu_samples.append(im)

    full_batches = len(cpu_samples) // batch
    if full_batches <= 0:
        raise RuntimeError(f"Not enough paired images for batch={batch}, only {len(cpu_samples)} samples found.")

    batches = []
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    for bi in range(full_batches):
        chunk = cpu_samples[bi * batch : (bi + 1) * batch]
        b = torch.stack(chunk, dim=0).contiguous()
        if half and use_cuda:
            b = b.half()
        b = b.to(device, non_blocking=use_cuda)
        batches.append(b)
    return batches, len(use_pairs), x_modality, xch


def benchmark_paper_forward(model, batches, warmup, steps, device):
    import torch

    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    batch = int(batches[0].shape[0])

    model.eval()
    ms_list = []
    with torch.inference_mode():
        for i in range(max(0, warmup)):
            xb = batches[i % len(batches)]
            if use_cuda:
                torch.cuda.synchronize(device)
            _ = model(xb)
            if use_cuda:
                torch.cuda.synchronize(device)

        for i in range(max(1, steps)):
            xb = batches[(i + warmup) % len(batches)]
            if use_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(xb)
            if use_cuda:
                torch.cuda.synchronize(device)
            ms_list.append((time.perf_counter() - t0) * 1000.0)

    mean_batch_ms = sum(ms_list) / len(ms_list)
    mean_img_ms = mean_batch_ms / batch
    fps = 1000.0 / mean_img_ms if mean_img_ms > 0 else 0.0
    return {
        "warmup_steps": max(0, warmup),
        "bench_steps": len(ms_list),
        "batch": batch,
        "latency_batch_ms": mean_batch_ms,
        "latency_img_ms": mean_img_ms,
        "fps": fps,
    }


def benchmark_e2e_predict(model, pairs, warmup, max_samples, device, imgsz, half, conf):
    total_need = min(len(pairs), max(1, warmup + max_samples))
    pairs = pairs[:total_need]

    use_cuda = str(device).startswith("cuda")
    total_ms_list = []
    for i, (rgb_p, x_p) in enumerate(pairs):
        if use_cuda:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = model.predict(
            rgb_source=str(rgb_p),
            x_source=str(x_p),
            device=str(device).replace("cuda:", ""),
            imgsz=imgsz,
            half=half,
            conf=conf,
            save=False,
            verbose=False,
            stream=False,
        )
        if use_cuda:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if not results or i < warmup:
            continue
        total_ms_list.append((time.perf_counter() - t0) * 1000.0)

    if not total_ms_list:
        raise RuntimeError("No valid e2e timing samples collected.")
    mean_ms = sum(total_ms_list) / len(total_ms_list)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "warmup_steps": max(0, warmup),
        "bench_steps": len(total_ms_list),
        "latency_img_ms": mean_ms,
        "fps": fps,
    }


def resolve_weight_path(args):
    raw = str(args.weights).strip()
    if raw:
        return Path(raw).expanduser()

    name = str(args.model_name).strip()
    if not name:
        raise ValueError("No model name provided.")

    p = Path(name).expanduser()
    if p.suffix.lower() == ".pt":
        return p

    return Path("ResTest") / name / "weights" / "best.pt"


def _one_line(mode, weights_name, args, stats, x_modality=None, xch=None):
    extra = ""
    if mode == "paper":
        extra = f", x={x_modality}/{xch}ch"
    return (
        f"[{mode}] model={weights_name} split={args.split} device={args.device} "
        f"batch={stats.get('batch', 1)} warmup={stats['warmup_steps']} bench={stats['bench_steps']}{extra} "
        f"latency={stats['latency_img_ms']:.3f}ms/img fps={stats['fps']:.2f}"
    )


def print_report(args, weights, rgb_dir, x_dir, pair_count, paper_stats=None, e2e_stats=None, x_modality=None, xch=None):
    weights = Path(weights)
    print("RTDETRMM FPS Benchmark")
    print(f"Weights file : {weights.name}")
    print(f"Weights path : {weights}")
    print(f"Split        : {args.split}")
    print(f"Device       : {args.device}")
    print(f"RGB dir      : {rgb_dir}")
    print(f"X dir        : {x_dir}")
    print(f"Paired images: {pair_count}")
    print("-" * 90)

    if paper_stats is not None:
        print(_one_line("paper", weights.name, args, paper_stats, x_modality=x_modality, xch=xch))
        print(f"PAPER_FPS={paper_stats['fps']:.2f} | PAPER_LATENCY_MS={paper_stats['latency_img_ms']:.3f}")

    if e2e_stats is not None:
        print(_one_line("e2e", weights.name, args, e2e_stats))
        print(f"E2E_FPS={e2e_stats['fps']:.2f} | E2E_LATENCY_MS={e2e_stats['latency_img_ms']:.3f}")


def append_report_txt(path, args, weights_name, paper_stats=None, e2e_stats=None, x_modality=None, xch=None):
    lines = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    if paper_stats is not None:
        lines.append(
            f"{ts}\tpaper\t{weights_name}\t{args.split}\t{args.device}\t{paper_stats['batch']}\t"
            f"{paper_stats['warmup_steps']}\t{paper_stats['bench_steps']}\t{x_modality}/{xch}ch\t"
            f"{paper_stats['latency_img_ms']:.3f}\t{paper_stats['fps']:.2f}\n"
        )

    if e2e_stats is not None:
        lines.append(
            f"{ts}\te2e\t{weights_name}\t{args.split}\t{args.device}\t1\t"
            f"{e2e_stats['warmup_steps']}\t{e2e_stats['bench_steps']}\t-\t"
            f"{e2e_stats['latency_img_ms']:.3f}\t{e2e_stats['fps']:.2f}\n"
        )

    if lines:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.writelines(lines)


def main():
    args = parse_args()
    import torch
    from ultralytics import RTDETRMM

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")

    pairs, rgb_dir, x_dir = collect_pairs(data_yaml, args.split)
    if not pairs:
        raise RuntimeError(f"No paired images found under:\nRGB: {rgb_dir}\nX:   {x_dir}")

    with open(data_yaml, "r", encoding="utf-8", errors="ignore") as f:
        data_cfg = yaml.safe_load(f) or {}
    x_modality_hint = _detect_x_modality(data_cfg)

    device = _resolve_device(args.device, torch)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    if args.half and device.type != "cuda":
        print("Warning: --half is CUDA-only, force disabled on non-CUDA device.")
        args.half = False

    weights = resolve_weight_path(args)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model_wrapper = RTDETRMM(str(weights))
    model = model_wrapper.model.to(device).eval()
    if args.half and device.type == "cuda":
        model.half()

    paper_stats = None
    e2e_stats = None
    x_modality = None
    xch = None

    if args.mode in {"paper", "both"}:
        batches, used_images, x_modality, xch = build_batches_for_paper(
            model=model,
            pairs=pairs,
            imgsz=args.imgsz,
            batch=args.batch,
            warmup=args.warmup,
            steps=args.max_samples,
            device=device,
            half=args.half,
            x_modality_hint=x_modality_hint,
        )
        if used_images < args.batch:
            raise RuntimeError(f"Not enough paired images: {used_images}, required >= batch({args.batch}).")
        paper_stats = benchmark_paper_forward(
            model=model,
            batches=batches,
            warmup=args.warmup,
            steps=args.max_samples,
            device=device,
        )

    if args.mode in {"e2e", "both"}:
        e2e_stats = benchmark_e2e_predict(
            model=model_wrapper,
            pairs=pairs,
            warmup=args.warmup,
            max_samples=args.max_samples,
            device=device,
            imgsz=args.imgsz,
            half=args.half,
            conf=args.conf,
        )

    print_report(
        args=args,
        weights=weights,
        rgb_dir=rgb_dir,
        x_dir=x_dir,
        pair_count=len(pairs),
        paper_stats=paper_stats,
        e2e_stats=e2e_stats,
        x_modality=x_modality,
        xch=xch,
    )

    if str(args.save_txt).strip():
        append_report_txt(
            path=args.save_txt,
            args=args,
            weights_name=Path(weights).name,
            paper_stats=paper_stats,
            e2e_stats=e2e_stats,
            x_modality=x_modality,
            xch=xch,
        )
        print(f"Saved: {args.save_txt}")


if __name__ == "__main__":
    main()
