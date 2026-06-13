import gc
import os
import warnings

warnings.filterwarnings("ignore", message=".*deterministic.*")
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


TEACHER_SPECS = [
    {
        "yaml": r"ultralytics/cfg/models/rtmm/r50/rtdetr-r50-mm-mid.yaml",
        "run_name": "teacher-r50-mm-mid",
        "batch": 4,
        "epochs": 150,
        "amp": True,
    },
    {
        "yaml": r"ultralytics/cfg/models/rtmm/r50/rtdetr-r50-mm-mid-aifi-dattention.yaml",
        "run_name": "teacher-r50-mm-mid-aifi-dattention",
        "batch": 4,
        "epochs": 150,
        "amp": True,
    },
]

DATA = r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml"
PROJECT = r"D:/JiQI/MM-experiment/TeacherRTDETRMM"
DEVICE = 0
WORKERS = 8


def release_cuda(torch_module) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        try:
            torch_module.cuda.ipc_collect()
        except Exception:
            pass


def main():
    print("[boot] trainRT_teachers.py started", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")
    os.environ.setdefault("YOLO_MM_QUIET_PROGRESS", "true")

    import torch
    from ultralytics import RTDETRMM

    for idx, spec in enumerate(TEACHER_SPECS, start=1):
        print(
            f"[train] start teacher {idx}/{len(TEACHER_SPECS)} "
            f"name={spec['run_name']} batch={spec['batch']}",
            flush=True,
        )
        model = RTDETRMM(spec["yaml"])
        model.train(
            data=DATA,
            epochs=spec["epochs"],
            device=DEVICE,
            batch=spec["batch"],
            amp=spec["amp"],
            deterministic=False,
            project=PROJECT,
            name=spec["run_name"],
            resume=False,
            workers=WORKERS,
        )

        del model
        release_cuda(torch)


if __name__ == "__main__":
    main()
