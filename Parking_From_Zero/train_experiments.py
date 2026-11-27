"""Run four controlled YOLO11n parking-occupancy experiments.

Every experiment starts from the same official pretrained yolo11n.pt checkpoint
and uses the same train/val/test split. The script does not use the old project
checkpoint or modify the source dataset.
"""
from pathlib import Path
import torch
from ultralytics import YOLO

WORK_DIR = Path("/Users/alikhansagat/Desktop/Ali/DeepLearning/Parking_From_Zero")
DATA_YAML = WORK_DIR / "yolo_dataset" / "dataset.yaml"
RUNS_DIR = WORK_DIR / "runs"

# Same training conditions for every experiment. AdamW is explicit so `lr0`
# is actually honored by Ultralytics instead of being selected automatically.
COMMON = {
    "data": str(DATA_YAML),
    "epochs": 15,
    "batch": 4,
    "seed": 1409,
    "deterministic": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "workers": 0,
    "patience": 15,
    "cache": False,
    "plots": True,
    "verbose": True,
}

# One baseline plus three one-at-a-time hyperparameter changes.
EXPERIMENTS = [
    {
        "name": "E1_baseline_640",
        "imgsz": 640,
        "batch": 4,
    },
    {
        "name": "E2_learning_rate_0005",
        "imgsz": 640,
        "batch": 4,
        "lr0": 0.0005,
    },
    {
        "name": "E3_batch_8",
        "imgsz": 640,
        "batch": 8,
    },
    {
        "name": "E4_image_size_960",
        "imgsz": 960,
        "batch": 4,
    },
]


def choose_device():
    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    device = choose_device()
    print(f"Device: {device}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Runs will be saved under: {RUNS_DIR}")
    print("Each run starts from official pretrained yolo11n.pt.")

    for experiment in EXPERIMENTS:
        name = experiment["name"]
        overrides = {key: value for key, value in experiment.items() if key != "name"}
        args = {
            **COMMON,
            **overrides,
            "device": device,
            "project": str(RUNS_DIR),
            "name": name,
        }

        print("\n" + "=" * 72)
        print(f"Starting {name}: imgsz={args['imgsz']}, "
              f"batch={args['batch']}, lr0={args['lr0']}, "
              f"optimizer={args['optimizer']}")
        print("=" * 72)

        # Recreate the same pretrained starting point for a fair comparison.
        model = YOLO("yolo11n.pt")
        model.train(**args)

    print("\nAll four runs finished.")
    print("Compare each run's results.csv and weights/best.pt under:", RUNS_DIR)


if __name__ == "__main__":
    main()
