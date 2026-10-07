"""
Build a merged-class copy of the Student Behaviour Detection dataset.

The source dataset treats "Using_phone" (class 0) and "phone" (class 5) as
two separate classes even though they're the same real-world behaviour.
Splitting the signal between them hurts both classes' metrics (see
confusion matrix: phone mAP was the weakest of all 12 classes). This script
writes a NEW copy of the dataset (original is left untouched) with the two
merged into a single "phone" class and every other class id shifted down
to stay contiguous.

Usage:
    cd pi-deploy
    python prepare_dataset.py

Output:
    ./dataset/{train,valid,test}/{images,labels}/   (images are symlinked,
                                                       labels are rewritten)
    ./dataset/data.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

SOURCE_ROOT = (
    Path(__file__).parent.parent / "Student Behaviour Detection.v1i.yolov8 2"
).resolve()

OUTPUT_ROOT = (Path(__file__).parent / "dataset").resolve()

# Original 12 classes, in id order, as they appear in the source data.yaml.
ORIGINAL_NAMES = [
    "Using_phone", "bend", "book", "bow_head",
    "hand-raising", "phone", "raise_head", "reading",
    "sleep", "turn_head", "upright", "writing",
]

# Merge "Using_phone" (0) into "phone" (5). Every other class keeps its
# identity but gets renumbered once the merge removes one slot.
MERGE_INTO = {
    "Using_phone": "phone",
}


def build_class_mapping() -> tuple[list[str], dict[int, int]]:
    merged_name_for = {name: MERGE_INTO.get(name, name) for name in ORIGINAL_NAMES}
    new_names = sorted(set(merged_name_for.values()), key=ORIGINAL_NAMES.index)
    # keep original relative order, de-duplicated
    seen = []
    for name in merged_name_for.values():
        if name not in seen:
            seen.append(name)
    new_names = seen

    old_id_to_new_id = {}
    for old_id, old_name in enumerate(ORIGINAL_NAMES):
        new_name = merged_name_for[old_name]
        new_id = new_names.index(new_name)
        old_id_to_new_id[old_id] = new_id

    return new_names, old_id_to_new_id


def remap_labels(src_labels_dir: Path, dst_labels_dir: Path, mapping: dict[int, int]) -> None:
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    for label_file in src_labels_dir.glob("*.txt"):
        lines_out = []
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            old_id = int(parts[0])
            new_id = mapping[old_id]
            lines_out.append(" ".join([str(new_id), *parts[1:]]))
        (dst_labels_dir / label_file.name).write_text("\n".join(lines_out) + "\n")


def link_images(src_images_dir: Path, dst_images_dir: Path) -> None:
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    for img in src_images_dir.iterdir():
        dst = dst_images_dir / img.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(img.resolve(), dst)


def main():
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Source dataset not found at: {SOURCE_ROOT}")

    new_names, mapping = build_class_mapping()
    print(f"[INFO] {len(ORIGINAL_NAMES)} classes -> {len(new_names)} classes")
    for old_id, new_id in mapping.items():
        if ORIGINAL_NAMES[old_id] != new_names[new_id]:
            print(f"       merged '{ORIGINAL_NAMES[old_id]}' (id {old_id}) -> "
                  f"'{new_names[new_id]}' (id {new_id})")

    for split in ("train", "valid", "test"):
        src_split = SOURCE_ROOT / split
        if not src_split.exists():
            continue
        print(f"[INFO] Processing split: {split}")
        link_images(src_split / "images", OUTPUT_ROOT / split / "images")
        remap_labels(src_split / "labels", OUTPUT_ROOT / split / "labels", mapping)

    data_yaml = {
        "train": str(OUTPUT_ROOT / "train" / "images"),
        "val":   str(OUTPUT_ROOT / "valid" / "images"),
        "test":  str(OUTPUT_ROOT / "test" / "images"),
        "nc":    len(new_names),
        "names": new_names,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\n[INFO] Merged dataset written to: {OUTPUT_ROOT}")
    print(f"[INFO] data.yaml: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
