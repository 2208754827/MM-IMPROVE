import os
import gc
import warnings

warnings.filterwarnings("ignore", message=".*deterministic.*")
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


def main():
    print("[boot] trainRT.py started", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("YOLO_TQDM_RICH", "false")
    os.environ.setdefault("YOLO_VERBOSE", "true")
    os.environ.setdefault("YOLO_MM_QUIET_PROGRESS", "true")

    import torch
    from ultralytics import RTDETRMM  
    #

    model_name1 = "A-C3k2_Faster_CGLU"
    model_name2 = "A-C3Ghost"

    batch1 = 4
    batch2 = 4
    workers = 8
    amp1 = True
    amp2 = True  # reduce late-stage NaN collapse risk on AddFusion-like variants

    print(f"[train] start model1={model_name1}, batch={batch1}", flush=True)

    model = RTDETRMM(f"ultralytics/cfg/models/rtmm/r18/{model_name1}.yaml")
    model.train(
        data=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml",
        epochs=150,
        device=0,
        batch=batch1,
        
        amp=amp1,
        deterministic=False,
        project="D:/JiQI/MM-experiment/ResTest",
        name=model_name1,
        resume=False,
        workers=workers,
    )

    # Release memory before training the heavier second model.
    if "model" in locals():
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    print(f"[train] start model2={model_name2}, batch={batch2}", flush=True)
    model = RTDETRMM(f"ultralytics/cfg/models/rtmm/r18/{model_name2}.yaml")
    model.train(
        data=r"D:\BaiduNetdiskDownload\M3FD\M3FD_split\data.yaml",
        epochs=150,
        device=0,
        batch=batch2,
        
        amp=amp2,
        deterministic=False,
        project="D:/JiQI/MM-experiment/ResTest",
        name=model_name2,
        resume=False,
        workers=workers,
    )

if __name__ == "__main__":
     main()
