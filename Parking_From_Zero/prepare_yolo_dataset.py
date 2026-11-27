"""Build a clean YOLO11 parking dataset from the original CNR-EXT files.

Only reads Project_Parking source images and CSV files. Writes new output under
Parking_From_Zero/yolo_dataset.
"""
from collections import defaultdict
from csv import DictReader, DictWriter
from pathlib import Path
import random
import re
import shutil

RAW_DIR = Path("/Users/alikhansagat/Desktop/Ali/DeepLearning/Project_Parking")
SOURCE_ROOT = RAW_DIR / "CNR-EXT_FULL_IMAGE_1000x750"
IMAGE_ROOT = SOURCE_ROOT / "FULL_IMAGE_1000x750"
WORK_DIR = Path("/Users/alikhansagat/Desktop/Ali/DeepLearning/Parking_From_Zero")
OUTPUT_DIR = WORK_DIR / "yolo_dataset"
OCCUPANCY_CSV = RAW_DIR / "CNRPark+EXT.csv"
SEED = 1409
TOTAL_FRAMES = 1600
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 750
SOURCE_WIDTH = 2592
SOURCE_HEIGHT = 1944


def camera_number(value):
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot read camera number from {value!r}")
    return int(match.group(1))


def slot_number(value):
    return str(int(float(value)))


def read_geometry():
    geometry = {}
    for camera_csv in sorted(SOURCE_ROOT.glob("camera*.csv")):
        cam = camera_number(camera_csv.stem)
        with camera_csv.open(newline="", encoding="utf-8-sig") as f:
            rows = list(DictReader(f))
        geometry[cam] = {slot_number(row["SlotId"]): row for row in rows}
    return geometry


def read_occupancy():
    by_frame = defaultdict(dict)
    with OCCUPANCY_CSV.open(newline="", encoding="utf-8-sig") as f:
        for row in DictReader(f):
            try:
                cam = camera_number(row["camera"])
            except ValueError:
                # The combined metadata CSV also includes the separate A/B subset.
                # Full-scene CNR-EXT frames use cameras 1 through 9.
                continue
            key = (cam, row["datetime"].strip())
            slot = slot_number(row["slot_id"])
            label = int(float(row["occupancy"]))
            if label not in (0, 1):
                continue
            by_frame[key][slot] = label
    return by_frame


def frame_key(image_path):
    cam = camera_number(image_path.parent.name)
    date, time = image_path.stem.rsplit("_", 1)
    # Raw image: 2015-11-16_0710; occupancy CSV: 2015-11-16_07.10
    timestamp = f"{date}_{time[:2]}.{time[2:]}"
    return cam, timestamp


def yolo_rows(image_path, camera_geometry, frame_status):
    cam, _ = frame_key(image_path)
    geometry = camera_geometry[cam]
    if set(geometry) - set(frame_status):
        return None

    sx = IMAGE_WIDTH / SOURCE_WIDTH
    sy = IMAGE_HEIGHT / SOURCE_HEIGHT
    rows = []
    for slot, box in geometry.items():
        label = frame_status[slot]
        x = float(box["X"]) * sx
        y = float(box["Y"]) * sy
        w = float(box["W"]) * sx
        h = float(box["H"]) * sy
        x1 = max(0.0, min(IMAGE_WIDTH, x))
        y1 = max(0.0, min(IMAGE_HEIGHT, y))
        x2 = max(0.0, min(IMAGE_WIDTH, x + w))
        y2 = max(0.0, min(IMAGE_HEIGHT, y + h))
        if x2 <= x1 or y2 <= y1:
            return None
        cx = ((x1 + x2) / 2) / IMAGE_WIDTH
        cy = ((y1 + y2) / 2) / IMAGE_HEIGHT
        nw = (x2 - x1) / IMAGE_WIDTH
        nh = (y2 - y1) / IMAGE_HEIGHT
        rows.append(f"{label} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return rows


def allocate_quotas(capacities, total):
    weights = {"train": 0.70, "val": 0.15, "test": 0.15}
    quotas = {name: int(total * weight) for name, weight in weights.items()}
    quotas["train"] += total - sum(quotas.values())
    for name in quotas:
        quotas[name] = min(quotas[name], capacities[name])
    while sum(quotas.values()) < total:
        available = [name for name in quotas if quotas[name] < capacities[name]]
        if not available:
            raise ValueError("Not enough complete labeled frames for requested total")
        name = max(available, key=lambda n: capacities[n] - quotas[n])
        quotas[name] += 1
    return quotas


def main():
    if not IMAGE_ROOT.is_dir() or not OCCUPANCY_CSV.is_file():
        raise FileNotFoundError("Original CNR-EXT images or occupancy CSV not found")

    camera_geometry = read_geometry()
    occupancy = read_occupancy()
    images = sorted(p for p in IMAGE_ROOT.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    candidates = []
    missing = 0
    incomplete = 0
    for image_path in images:
        cam, timestamp = frame_key(image_path)
        status = occupancy.get((cam, timestamp))
        if not status:
            missing += 1
            continue
        labels = yolo_rows(image_path, camera_geometry, status)
        if labels is None:
            incomplete += 1
            continue
        date = image_path.stem.split("_", 1)[0]
        candidates.append({"image": image_path, "camera": cam, "date": date, "labels": labels})

    if len(candidates) < TOTAL_FRAMES:
        raise ValueError(f"Only {len(candidates)} complete frames; need {TOTAL_FRAMES}")

    # Keep all frames from a calendar date in one split to limit near-duplicate leakage.
    dates = sorted({item["date"] for item in candidates})
    rng = random.Random(SEED)
    rng.shuffle(dates)
    n_train = max(1, round(len(dates) * 0.70))
    n_val = max(1, round(len(dates) * 0.15))
    date_splits = {
        "train": set(dates[:n_train]),
        "val": set(dates[n_train:n_train + n_val]),
        "test": set(dates[n_train + n_val:]),
    }
    by_split = {
        split: [item for item in candidates if item["date"] in split_dates]
        for split, split_dates in date_splits.items()
    }
    quotas = allocate_quotas({k: len(v) for k, v in by_split.items()}, TOTAL_FRAMES)

    selected = {}
    for split, items in by_split.items():
        rng.shuffle(items)
        selected[split] = items[:quotas[split]]

    manifest_path = OUTPUT_DIR / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for split, items in selected.items():
        image_dir = OUTPUT_DIR / "images" / split
        label_dir = OUTPUT_DIR / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            source = item["image"]
            basename = f"camera{item['camera']:02d}_{source.stem}.jpg"
            target_image = image_dir / basename
            target_label = label_dir / f"{Path(basename).stem}.txt"
            shutil.copy2(source, target_image)
            target_label.write_text("\n".join(item["labels"]) + "\n", encoding="utf-8")
            busy = sum(line.startswith("1 ") for line in item["labels"])
            manifest_rows.append({
                "split": split,
                "image": str(target_image.relative_to(OUTPUT_DIR)),
                "date": item["date"],
                "camera": item["camera"],
                "parking_spaces": len(item["labels"]),
                "free": len(item["labels"]) - busy,
                "occupied": busy,
            })

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=["split", "image", "date", "camera", "parking_spaces", "free", "occupied"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    yaml_text = (
        f"path: {OUTPUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: free\n"
        "  1: occupied\n"
    )
    (OUTPUT_DIR / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Исходных кадров найдено: {len(images)}")
    print(f"Кадров без статусов: {missing}; неполная разметка: {incomplete}")
    print(f"Полных размеченных кадров: {len(candidates)}")
    print(f"Сохранено кадров: {len(manifest_rows)}")
    print(f"Train/val/test: {quotas['train']}/{quotas['val']}/{quotas['test']}")
    print(f"Дата-групп train/val/test: {len(date_splits['train'])}/{len(date_splits['val'])}/{len(date_splits['test'])}")
    print(f"Разметка и конфиг: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
