import os
import warnings

warnings.filterwarnings("ignore", message=".*deterministic.*")
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("YOLO_MM_QUIET_PROGRESS", "true")

    from ultralytics import RTDETRMM

    # 从头训练 baseline_plus_csp
    model = RTDETRMM(
        r"D:\BaiduNetdiskDownload\MutilModel_3398475911\ultralytics\cfg\models\rtmm\r18\baseline_plus_cmx.yaml"
    )

    model.train(
        data=r"D:\BaiduNetdiskDownload\M3FD_split\data.yaml",
        epochs=250,
        imgsz=640,
        batch=4,
        device=0,
        workers=8,

        optimizer="auto",
        lr0=0.01,
        cos_lr=False,

        amp=False,
        deterministic=False,
        save_period=-1,
        project=r"D:\JiQI\MM-experiment\ResTest",
        name="only-asf",
        exist_ok=False,
        resume=False,
    )


if __name__ == "__main__":
    main()

