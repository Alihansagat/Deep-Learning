"""Export per-crop held-out test predictions for evidence and error analysis."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import (ID_TO_LABEL, ParkingPatchDataset, SmallParkingCNN,
                   choose_device, discover_samples, make_splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("CNRPark-Patches-150x150"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/final_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=1409)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = choose_device("auto")
    samples = discover_samples(args.data_dir)
    _, _, test, _ = make_splits(samples, args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = SmallParkingCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(ParkingPatchDataset(test, checkpoint["image_size"]),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    cursor = 0
    with torch.no_grad():
        for images, _labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            for prob in probs:
                sample = test[cursor]
                pred = int(np.argmax(prob))
                rows.append({
                    "image": sample.path.name,
                    "true_label": ID_TO_LABEL[sample.label],
                    "predicted_label": ID_TO_LABEL[pred],
                    "confidence": float(prob[pred]),
                    "prob_busy": float(prob[0]),
                    "prob_free": float(prob[1]),
                    "correct": pred == sample.label,
                    "camera": sample.camera,
                    "date": sample.date,
                    "time": sample.time_key,
                })
                cursor += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    errors = [row for row in rows if not row["correct"]]
    # If fewer than three errors exist, surface the most uncertain correct cases,
    # as required by the coursework's instructions for a low-error test set.
    if len(errors) >= 3:
        # Include both types of mistakes, prioritizing cases near the decision
        # boundary so the visual reason is easier to discuss in the defense.
        missed_busy = sorted((r for r in errors if r["true_label"] == "busy"), key=lambda r: r["confidence"])
        false_busy = sorted((r for r in errors if r["true_label"] == "free"), key=lambda r: r["confidence"])
        selected = missed_busy[:2] + false_busy[:1]
    else:
        selected = sorted(rows, key=lambda r: abs(r["prob_free"] - 0.5))[:3]
    for item in selected:
        if item["true_label"] == "busy" and item["predicted_label"] == "free":
            item["analysis_note"] = "False-free prediction: foliage, shadows, or partial vehicle occlusion can hide the occupied-space cues in this crop."
        elif item["true_label"] == "free" and item["predicted_label"] == "busy":
            item["analysis_note"] = "False-busy prediction: a neighboring vehicle or wheel can enter the crop even though the target space is labeled free."
        else:
            item["analysis_note"] = "Challenging correct example near the decision threshold; inspect for shadows, occlusion, and camera-view changes."
    payload = {
        "test_examples_total": len(rows),
        "test_errors_total": len(errors),
        "analysis_mode": "misclassifications" if len(errors) >= 3 else "hardest examples by confidence (fewer than three errors)",
        "selected_cases": selected,
        "next_step": "Collect or add labeled samples under the new camera/viewpoint and evaluate again using a new held-out set.",
    }
    (args.output_dir / "failure_cases.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows; errors={len(errors)}; case analysis={payload['analysis_mode']}")


if __name__ == "__main__":
    main()
