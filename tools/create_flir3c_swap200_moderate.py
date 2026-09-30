from __future__ import annotations

import shutil
from pathlib import Path

SOURCE = Path(r"D:\BaiduNetdiskDownload\FLIR_mm")
TARGET = Path(r"D:\BaiduNetdiskDownload\FLIR3C\FLIR3C_swap200_moderate")
STAGING = Path(r"D:\BaiduNetdiskDownload\FLIR3C\FLIR3C_swap200_moderate_staging")
MANIFEST_SOURCE = Path(r"D:\BaiduNetdiskDownload\MutilModel_3398475911\datasets\flir_custom_swap200_moderate")


def read_names(filename: str) -> list[str]:
    names = [line.strip() for line in (MANIFEST_SOURCE / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(names) != 200 or len(set(names)) != 200:
        raise RuntimeError(f"{filename}: expected 200 unique names, got {len(names)}/{len(set(names))}")
    return names


def frame_id(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[1])


def leakage(train_names: set[str], test_names: set[str]) -> dict[int, int]:
    train_ids = {frame_id(name) for name in train_names}
    test_ids = [frame_id(name) for name in test_names]
    return {
        distance: sum(
            any(test_id + offset in train_ids for offset in range(-distance, distance + 1))
            for test_id in test_ids
        )
        for distance in (0, 1, 2, 5, 10)
    }


def class_counts(names: set[str], official_train: set[str]) -> tuple[list[int], list[int]]:
    instances = [0, 0, 0]
    images = [0, 0, 0]
    for name in names:
        source_split = "train" if name in official_train else "test"
        label_path = SOURCE / "labels" / source_split / Path(name).with_suffix(".txt").name
        seen: set[int] = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0])
            if 0 <= cls < 3:
                instances[cls] += 1
                seen.add(cls)
        for cls in seen:
            images[cls] += 1
    return instances, images


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {TARGET}")
    if STAGING.exists():
        raise FileExistsError(f"Staging path already exists: {STAGING}")

    official_train = {p.name for p in (SOURCE / "images" / "train").glob("*.jpg")}
    official_test = {p.name for p in (SOURCE / "images" / "test").glob("*.jpg")}
    train_to_test = set(read_names("official_train_to_custom_test_200.txt"))
    test_to_train = set(read_names("official_test_to_custom_train_200.txt"))

    if not train_to_test <= official_train:
        raise RuntimeError("train-to-test manifest contains names outside official train")
    if not test_to_train <= official_test:
        raise RuntimeError("test-to-train manifest contains names outside official test")

    custom_train = (official_train - train_to_test) | test_to_train
    custom_test = (official_test - test_to_train) | train_to_test
    if len(custom_train) != 4129 or len(custom_test) != 1013 or custom_train & custom_test:
        raise RuntimeError("Invalid custom split counts or overlap")

    leak = leakage(custom_train, custom_test)
    expected_leak = {0: 0, 1: 98, 2: 187, 5: 347, 10: 548}
    if leak != expected_leak:
        raise RuntimeError(f"Unexpected leakage result: {leak}, expected {expected_leak}")

    try:
        for modality in ("images", "images_ir", "labels"):
            for split in ("train", "test"):
                (STAGING / modality / split).mkdir(parents=True, exist_ok=True)

        for destination_split, names in (("train", custom_train), ("test", custom_test)):
            for name in sorted(names):
                source_split = "train" if name in official_train else "test"
                for modality, suffix in (("images", ".jpg"), ("images_ir", ".jpg"), ("labels", ".txt")):
                    actual_name = Path(name).with_suffix(suffix).name
                    source_file = SOURCE / modality / source_split / actual_name
                    destination_file = STAGING / modality / destination_split / actual_name
                    if not source_file.is_file():
                        raise FileNotFoundError(source_file)
                    shutil.copy2(source_file, destination_file)

        yaml_text = """# FLIR RGB+IR custom split: exactly 200 images exchanged in each direction.
# Moderate frame-neighbor overlap; custom experiment only, not official FLIR evaluation.
path: D:/BaiduNetdiskDownload/FLIR3C/FLIR3C_swap200_moderate
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

        manifest_dir = STAGING / "split_manifest"
        manifest_dir.mkdir()
        for filename in (
            "official_train_to_custom_test_200.txt",
            "official_test_to_custom_train_200.txt",
            "search_summary.txt",
        ):
            shutil.copy2(MANIFEST_SOURCE / filename, manifest_dir / filename)
        (manifest_dir / "train_names.txt").write_text("\n".join(sorted(custom_train)) + "\n", encoding="utf-8")
        (manifest_dir / "test_names.txt").write_text("\n".join(sorted(custom_test)) + "\n", encoding="utf-8")

        train_stats = class_counts(custom_train, official_train)
        test_stats = class_counts(custom_test, official_train)
        summary = [
            "FLIR swap200 moderate-overlap custom split",
            "Generated: 2026-09-22",
            r"Source: D:\BaiduNetdiskDownload\FLIR_mm",
            "Official train -> custom test: 200",
            "Official test -> custom train: 200",
            "Custom train/test: 4129/1013",
            f"Leakage counts: {leak}",
            f"Leakage percentages: { {d: round(n / 1013 * 100, 2) for d, n in leak.items()} }",
            f"Train class counts (instances, images): {train_stats}",
            f"Test class counts (instances, images): {test_stats}",
            "This is a custom split and must not be reported as official FLIR evaluation.",
        ]
        (manifest_dir / "manifest.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

        for modality in ("images", "images_ir", "labels"):
            for split, expected in (("train", 4129), ("test", 1013)):
                actual = sum(1 for item in (STAGING / modality / split).iterdir() if item.is_file())
                if actual != expected:
                    raise RuntimeError(f"Count mismatch for {modality}/{split}: {actual} != {expected}")
        for split in ("train", "test"):
            rgb = {p.stem for p in (STAGING / "images" / split).glob("*.jpg")}
            ir = {p.stem for p in (STAGING / "images_ir" / split).glob("*.jpg")}
            labels = {p.stem for p in (STAGING / "labels" / split).glob("*.txt")}
            if rgb != ir or rgb != labels:
                raise RuntimeError(f"RGB/IR/label mismatch in {split}")

        STAGING.rename(TARGET)
    except Exception:
        # Preserve staging on failure for diagnosis. Existing datasets are never changed.
        raise

    print(f"CREATED: {TARGET}")
    print(f"train/test: {len(custom_train)}/{len(custom_test)}")
    print(f"leakage: {leak}")
    print(f"leakage_pct: { {d: round(n / 1013 * 100, 2) for d, n in leak.items()} }")
    print(f"test_class_counts: {class_counts(custom_test, official_train)}")


if __name__ == "__main__":
    main()
