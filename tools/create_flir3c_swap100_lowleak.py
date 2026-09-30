from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SOURCE = Path(r"D:\BaiduNetdiskDownload\FLIR_mm")
TARGET = Path(r"D:\BaiduNetdiskDownload\FLIR3C\FLIR3C_swap100_lowleak")
STAGING = Path(r"D:\BaiduNetdiskDownload\FLIR3C\FLIR3C_swap100_lowleak_staging")
THRESHOLD = 5

# Whole frame-components are exchanged rather than isolated frames. Every component
# is separated from the remaining images in its original split by more than 5 IDs.
TRAIN_COMPONENT_INDEXES = (63, 72, 184)       # 29 + 48 + 23 = 100 images
TEST_COMPONENT_INDEXES = (0, 1, 8, 11, 13)   # 69 + 8 + 13 + 3 + 7 = 100 images


def image_ids(split: str) -> list[int]:
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in (SOURCE / "images" / split).glob("*.jpg"))


def components(ids: list[int], threshold: int = THRESHOLD) -> list[list[int]]:
    result: list[list[int]] = []
    current = [ids[0]]
    for previous, current_id in zip(ids, ids[1:]):
        if current_id - previous > threshold:
            result.append(current)
            current = [current_id]
        else:
            current.append(current_id)
    result.append(current)
    return result


def file_name(frame_id: int) -> str:
    return f"FLIR_{frame_id:05d}.jpg"


def selected_names(split: str, indexes: tuple[int, ...]) -> list[str]:
    comps = components(image_ids(split))
    names = [file_name(frame_id) for index in indexes for frame_id in comps[index]]
    if len(names) != 100:
        raise RuntimeError(f"Expected exactly 100 selected {split} images, got {len(names)}")
    return sorted(names)


def leakage_counts(train_names: set[str], test_names: set[str]) -> dict[int, int]:
    train_ids = {int(Path(name).stem.rsplit("_", 1)[1]) for name in train_names}
    test_ids = [int(Path(name).stem.rsplit("_", 1)[1]) for name in test_names]
    counts = {}
    for distance in (0, 1, 2, 5, 10):
        counts[distance] = sum(
            any(frame_id + offset in train_ids for offset in range(-distance, distance + 1))
            for frame_id in test_ids
        )
    return counts


def class_counts(split_names: dict[str, set[str]]) -> dict[str, dict[str, list[int]]]:
    output = {}
    official_train = {p.name for p in (SOURCE / "images" / "train").glob("*.jpg")}
    for split, names in split_names.items():
        instances = [0, 0, 0]
        images = [0, 0, 0]
        for name in names:
            source_split = "train" if name in official_train else "test"
            label = SOURCE / "labels" / source_split / Path(name).with_suffix(".txt").name
            seen = set()
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls = int(line.split()[0])
                if 0 <= cls < 3:
                    instances[cls] += 1
                    seen.add(cls)
            for cls in seen:
                images[cls] += 1
        output[split] = {"instances": instances, "images": images}
    return output


def build_split() -> tuple[dict[str, set[str]], list[str], list[str]]:
    official_train = {p.name for p in (SOURCE / "images" / "train").glob("*.jpg")}
    official_test = {p.name for p in (SOURCE / "images" / "test").glob("*.jpg")}
    train_to_test = selected_names("train", TRAIN_COMPONENT_INDEXES)
    test_to_train = selected_names("test", TEST_COMPONENT_INDEXES)

    custom_train = (official_train - set(train_to_test)) | set(test_to_train)
    custom_test = (official_test - set(test_to_train)) | set(train_to_test)
    if len(custom_train) != 4129 or len(custom_test) != 1013 or custom_train & custom_test:
        raise RuntimeError("Invalid custom split counts or overlap")
    return {"train": custom_train, "test": custom_test}, train_to_test, test_to_train


def copy_dataset(split_names: dict[str, set[str]], train_to_test: list[str], test_to_train: list[str]) -> None:
    if TARGET.exists():
        raise FileExistsError(f"Target already exists; refusing to overwrite: {TARGET}")
    if STAGING.exists():
        raise FileExistsError(f"Staging already exists; remove it manually after checking: {STAGING}")

    official_train = {p.name for p in (SOURCE / "images" / "train").glob("*.jpg")}
    try:
        for modality in ("images", "images_ir", "labels"):
            for split in ("train", "test"):
                (STAGING / modality / split).mkdir(parents=True, exist_ok=True)

        for destination_split, names in split_names.items():
            for name in sorted(names):
                source_split = "train" if name in official_train else "test"
                for modality, suffix in (("images", ".jpg"), ("images_ir", ".jpg"), ("labels", ".txt")):
                    actual_name = Path(name).with_suffix(suffix).name
                    source_file = SOURCE / modality / source_split / actual_name
                    destination_file = STAGING / modality / destination_split / actual_name
                    if not source_file.is_file():
                        raise FileNotFoundError(source_file)
                    shutil.copy2(source_file, destination_file)

        yaml_text = """# FLIR RGB+IR custom split: 100 whole frame-components exchanged in each direction.
# Selection preserves a >5 frame-ID gap across train/test; this is still a custom, non-official split.
path: D:/BaiduNetdiskDownload/FLIR3C/FLIR3C_swap100_lowleak
train: ./images/train
val: ./images/test
test: ./images/test

modality:
  rgb: images
  ir: images_ir

modality_used: ['rgb', 'ir']
Xch: 3
nc: 3
names:
  0: person
  1: car
  2: bicycle
"""
        (STAGING / "data.yaml").write_text(yaml_text, encoding="utf-8")
        manifest = STAGING / "split_manifest"
        manifest.mkdir()
        (manifest / "official_train_to_custom_test_100.txt").write_text("\n".join(train_to_test) + "\n", encoding="utf-8")
        (manifest / "official_test_to_custom_train_100.txt").write_text("\n".join(test_to_train) + "\n", encoding="utf-8")
        for split in ("train", "test"):
            (manifest / f"{split}_names.txt").write_text("\n".join(sorted(split_names[split])) + "\n", encoding="utf-8")

        leakage = leakage_counts(split_names["train"], split_names["test"])
        stats = class_counts(split_names)
        summary = [
            "FLIR swap100 low-leak custom split",
            "Generated: 2026-09-22",
            "Source: D:\\BaiduNetdiskDownload\\FLIR_mm",
            "Method: exchange complete connected frame-ID components using threshold <=5.",
            "Official train -> custom test: 100",
            "Official test -> custom train: 100",
            "Custom train/test: 4129/1013",
            f"Leakage counts: {leakage}",
            f"Class counts: {stats}",
            "This remains a custom split and must not be reported as official FLIR evaluation.",
        ]
        (manifest / "manifest.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

        for modality in ("images", "images_ir", "labels"):
            for split, expected in (("train", 4129), ("test", 1013)):
                actual = sum(1 for p in (STAGING / modality / split).iterdir() if p.is_file())
                if actual != expected:
                    raise RuntimeError(f"Count mismatch {modality}/{split}: {actual} != {expected}")
        for split in ("train", "test"):
            rgb = {p.stem for p in (STAGING / "images" / split).glob("*.jpg")}
            ir = {p.stem for p in (STAGING / "images_ir" / split).glob("*.jpg")}
            labels = {p.stem for p in (STAGING / "labels" / split).glob("*.txt")}
            if rgb != ir or rgb != labels:
                raise RuntimeError(f"RGB/IR/label mismatch in {split}")

        STAGING.rename(TARGET)
    except Exception:
        # Keep a failed staging directory for diagnosis; never touch existing datasets.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Copy files and create the new dataset")
    args = parser.parse_args()

    split_names, train_to_test, test_to_train = build_split()
    leakage = leakage_counts(split_names["train"], split_names["test"])
    stats = class_counts(split_names)
    print(f"train_to_test={len(train_to_test)}, test_to_train={len(test_to_train)}")
    print(f"custom_train={len(split_names['train'])}, custom_test={len(split_names['test'])}")
    print(f"leakage={leakage}")
    print(f"class_counts={stats}")
    print("train components:", [(train_to_test[0], train_to_test[-1])])
    print("test components:", [(test_to_train[0], test_to_train[-1])])
    if args.create:
        copy_dataset(split_names, train_to_test, test_to_train)
        print(f"CREATED: {TARGET}")
    else:
        print("DRY RUN ONLY: pass --create to build the dataset")


if __name__ == "__main__":
    main()
