# Parking Space Occupancy Object Detection

**Course:** CSCI1409 Deep Learning — Bonus Project
**Task:** detect each parking space in a full parking-lot image and classify it as `free` or `busy`.
**Model:** pretrained YOLO11n fine-tuned on CNRPark+EXT.
**Data:** 1,600 reproducibly selected full frames; date-grouped train/validation/test split.

## Problem

Drivers waste time looking for parking. This prototype analyzes one fixed-camera image and draws a box around every marked parking space, labeling it free or occupied. It is an offline research prototype trained on a public Italian parking lot; it has not been tested at our university, and its boxes are tied to the dataset's camera layouts.

## Dataset and labels

The project combines three supplied CNRPark+EXT assets:

1. `CNR-EXT_FULL_IMAGE_1000x750/`: full-scene images (4,278 frames).
2. `CNRPark+EXT.csv`: occupancy state by camera, timestamp, and slot ID.
3. `CNR-EXT_FULL_IMAGE_1000x750/cameraN.csv`: fixed parking-space coordinates by slot ID.

The geometry CSV coordinates use the original 2592×1944 image coordinate system; the released full frames are 1000×750. The notebook scales the boxes, joins state and geometry on camera/timestamp/slot ID, keeps only complete matches, and selects 1,600 frames with seed 1409. `occupancy=0` is mapped to `free`; `occupancy=1` to `busy`. The selection contains 57,213 labeled parking-space boxes. Raw images and the full CSV are not stored in Git.

Official dataset page and downloads: [CNR Parking Dataset](https://cnrpark.it/). The full-frame archive is about 1.1 GB; the source is licensed under ODbL 1.0. See the dataset page for attribution and license terms. The detector is [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11); the project documentation lists AGPL-3.0 for YOLO11 models, so review the license terms before redistributing the weights.

### Split policy

Frames are split by calendar date with `GroupShuffleSplit`, so frames from a given day cannot appear in multiple splits. This limits leakage from near-identical scenes captured on the same day.

| Split | Frames | Parking-space boxes | Date groups |
|---|---:|---:|---:|
| Train | 965 | 34,704 | 14 |
| Validation | 288 | 10,371 | 4 |
| Test | 347 | 12,138 | 5 |

## Model and controlled experiment

Both runs fine-tune the same pretrained YOLO11n detector for 15 epochs, with seed 1409, batch size 8, and identical data and training settings. The only planned change is image size:

| Run | Image size |
|---|---:|
| Baseline | 640 |
| Experiment | 800 |

After each run, its best checkpoint is compared on **validation mAP@0.50**. The higher-scoring run is selected, then evaluated once on the held-out test split. Precision, recall, mAP@0.50, the experiment table, training curves, error table, and prediction examples display inline in the notebook. The notebook keeps only the selected weights as `parking_yolo11n.pt`; temporary training logs and checkpoints are removed after the displayed results are produced.

## Run locally

Use Python 3.10–3.13. Put the data assets above in the project root, preserving the full-frame directory layout. Open `1.ipynb` in VS Code and run its cells from top to bottom. The first code cell installs dependencies. To predict on another full-scene photo after training, set `IMAGE_TO_PREDICT` in the final code cell.

Training uses Apple MPS when available, then CUDA, then CPU. A GPU is recommended for faster training. The notebook downloads the official `yolo11n.pt` pretrained weights on first use. Inference on a campus image is only a demo; campus accuracy has not been evaluated.

## Results and interpretation

Validation results were used to select the image size. The test split was reported only for the selected 800-pixel model.

| Run | Image size | Val precision | Val recall | Val mAP@0.50 | Val mAP@0.50:0.95 | Test precision | Test recall | Test mAP@0.50 | Test mAP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 640 | 0.9861 | 0.9566 | 0.9911 | 0.7559 | — | — | — | — |
| Selected experiment | 800 | 0.9825 | 0.9691 | 0.9920 | 0.7859 | 0.9891 | 0.9734 | 0.9933 | 0.7865 |

The 800-pixel run was selected by validation mAP@0.50 (0.9920 vs 0.9911). In a separate visual audit of 24 of the 347 test frames at confidence 0.25 and IoU 0.50, the model had 5 missed slots, 70 extra boxes, and 3 wrong-class matches. This low confidence threshold surfaces candidate detections; the full test metrics above use Ultralytics' standard validation procedure. The notebook stores the metric table, training curves, error review, and prediction images as inline outputs.

## Limitations

- This is a dataset benchmark, not a live campus-parking system. We did not record or evaluate university-lot images.
- The source has fixed cameras and known slot geometry; camera movement or a different lot layout requires new calibration and evaluation.
- Nearby frames are highly similar. Grouping by date reduces temporal leakage, but the cameras and physical lot remain shared across splits.
- Boxes are derived from the dataset's fixed rectangular parking-space annotations, not manually drawn around individual vehicles.
- Results can vary by hardware, Ultralytics/PyTorch version, and training runtime. The notebook prints the active device and keeps the selected weights for the final submission.

## Repository contents

- `1.ipynb` — the single notebook with all project code and explanatory comments.
- `README.md` — project summary, run instructions, experiment design, and limitations.
- `parking_yolo11n.pt` — selected detector weights.

The student should be ready to explain the label join, grouped split, YOLO fine-tuning, image-size experiment, and limitations during the project presentation.
