# Parking Space Occupancy Detection — Four Controlled Experiments

**Author:** Alikhan Sagat · **Course:** CSCI1409 Deep Learning · Bonus Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alihansagat/Deep-Learning/blob/main/Project_Parking/1.ipynb)

Detect parking spaces in full-frame images and classify each as `free` or `busy`. The model returns boxes, class labels and confidence scores; counts are derived from those detections. This is a public-data research prototype. University parking-lot performance has not been evaluated.

## Dataset, source and terms

Source: [CNR Parking Dataset](https://cnrpark.it/), CNRPark+EXT, distributed under **ODbL 1.0**. The notebook credits the original source. The supplied release archive is a documented derived subset; the original license and attribution apply. The original full-frame archive contains 1000×750 camera images and camera-specific geometry CSVs. The occupancy CSV supplies free/busy states by camera, timestamp and space ID.

The preparation code joins these three keys, rescales coordinates from 2592×1944 to 1000×750, rejects frames without complete matching labels, and samples 1,600 frames using seed 1409. YOLO labels are `class_id center_x center_y width height`, with normalized coordinates; **0=free, 1=busy**. Bounding boxes represent parking spaces, including empty spaces, rather than individual vehicles.

There are 57,213 annotation instances across the 1,600 images. A physical space is observed repeatedly, so this is not a count of unique physical spaces.

| Split | Images | Free boxes | Busy boxes | Total boxes | Date groups |
|---|---:|---:|---:|---:|---:|
| Train | 965 | 16,459 | 18,245 | 34,704 | 14 |
| Validation | 288 | 3,074 | 7,297 | 10,371 | 4 |
| Test | 347 | 7,060 | 5,078 | 12,138 | 5 |

`GroupShuffleSplit` holds calendar dates apart (seed 1409); no date is shared between splits. Fractions refer to date groups, not exact frame percentages. Cameras and the physical lot are shared. All image/label pairs and normalized coordinates are checked. Exact cross-split duplicates are checked with SHA-256. Five images with ground-truth boxes are displayed inline.

[Download prepared subset (~423 MB)](https://github.com/Alihansagat/Deep-Learning/releases/download/parking-four-experiments-v1/parking_subset.zip)

Archive SHA-256: `79c3549de5eec4dffeb47857aff862cea0052bfdaf63001adde19cddd1ec28f4`.

The archive contains `data/parking_yolo/images`, labels, `manifest.csv` and dataset YAML. The manifest records source paths, camera, timestamp, date, split and space count. In Colab, the notebook downloads and extracts this exact subset and rewrites the dataset root path for the runtime. With the original raw assets present locally, the same notebook regenerates the subset instead.

## Model and experiment design

Model: **Ultralytics YOLO11n**, each run initialized independently from the same official `yolo11n.pt`. Architecture and pretrained weights are from [Ultralytics](https://docs.ultralytics.com/models/yolo11/); these weights are not an architecture developed from scratch. Ultralytics provides AGPL-3.0 and commercial licensing; consult the upstream license for your intended use.

Four runs: baseline plus three one-at-a-time changes. Data, seed, initial checkpoint, optimizer, epochs and augmentation remain fixed.

| Run | Image size | Initial LR | Batch | Change relative to E0 |
|---|---:|---:|---:|---|
| E0 baseline | 640 | 0.001 | 8 | None |
| E1 learning rate | 640 | 0.0003 | 8 | Learning rate |
| E2 image size | 800 | 0.001 | 8 | Image size |
| E3 batch size | 640 | 0.001 | 4 | Batch size |

Shared settings: AdamW, 10 epochs, seed 1409, deterministic mode, warmup 1 epoch, linear LR schedule ending at 10% of initial LR, Mosaic 1.0 and `close_mosaic=0`. `optimizer="AdamW"` is explicit because the auto optimizer may ignore user LR. `nbs=1` disables gradient accumulation and `weight_decay=0` prevents batch-dependent decay scaling in all runs. The smaller batch performs more updates per epoch, so this compares the same epoch budget rather than an equal number of optimizer steps.

Hypotheses specified before results: lower LR may stabilize fine-tuning but slow progress; larger images may preserve small-space detail; smaller batches may change generalization through noisier gradients and reduce memory demand. The same augmentation settings apply to every experiment; augmentation is not a tested variable.

Selection rule: choose the setting with highest **validation mAP@0.50:0.95**, then evaluate only that setting on test. Each run's best checkpoint is selected through validation by the framework. Runs are limited to 10 epochs and a single seed; do not interpret small differences as statistically significant.

## Results

| Run | Change | Val P | Val R | Val mAP50 | Val mAP50:95 | Training minutes |
|---|---|---:|---:|---:|---:|---:|
| E0_baseline | Baseline | 0.9759 | 0.9430 | 0.9868 | 0.7121 | 22.9 |
| E1_learning_rate | Learning rate | 0.9736 | 0.9350 | 0.9847 | 0.7084 | 27.5 |
| E2_image_size | Image size | 0.9756 | 0.9666 | 0.9917 | 0.7621 | 59.0 |
| E3_batch_size | Batch size | 0.9752 | 0.9490 | 0.9890 | 0.7124 | 20.9 |

Selected using validation mAP50:95: **E2_image_size**.

| Test metric | Value |
|---|---:|
| precision | 0.9835 |
| recall | 0.9715 |
| map50 | 0.9928 |
| map50_95 | 0.7607 |

Measured validation mAP50:95 changes relative to E0: E1_learning_rate -0.37 percentage points, E2_image_size +4.99 percentage points, E3_batch_size +0.03 percentage points.

The winner is chosen only on validation. These results describe this single-seed, 10-epoch budget; they do not establish which setting would be best after full convergence or on another parking lot.

## Evaluation settings and error analysis

Evaluation uses candidate confidence **0.001** and NMS IoU **0.7**. These are distinct from the correctness thresholds used by mAP. mAP@0.50 matches at IoU 0.50; mAP@0.50:0.95 averages across IoU thresholds from 0.50 to 0.95 in steps of 0.05. Class must also match. AP summarizes a precision–recall curve; mAP averages AP over both classes.

Precision is the fraction of predicted detections that are correct; recall is the fraction of ground-truth objects detected. The framework reports mean P/R at its F1-selected confidence threshold. These are not image classification accuracy, nor are they automatically measured at demo confidence 0.25.

Five test images from different cameras show boxes, names and confidence scores inline. The diagnostic error audit processes all 347 test frames at confidence 0.25, NMS IoU 0.7 and matching IoU 0.50. It greedily matches boxes by geometry and reports wrong-class matches separately; this diagnostic is not the standard mAP implementation. Three actual test failure cases are shown side by side with ground truth in the notebook.

At the diagnostic threshold: 187 missed spaces, 2052 unmatched extra boxes and 47 wrong-class matches across all 347 test frames. These counts use a different confidence setting and matching procedure from the standard P/R metrics.

| Failure example | Missed | Extra | Wrong class |
|---|---:|---:|---:|
| cam09_201511211717.jpg | 17 | 13 | 2 |
| cam09_201511211747.jpg | 21 | 7 | 2 |
| cam01_201601121217.jpg | 7 | 15 | 1 |

## Setup and run

1. Open the Colab link and choose **Runtime → Change runtime type → GPU**. A paid runtime is not required, but training duration depends on available compute. Alternatively open `1.ipynb` in VS Code from this project directory.
2. Run cells from top to bottom. The installation cell pins Ultralytics 8.4.218, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, Matplotlib 3.10.7, Pillow 12.0.0 and PyYAML 6.0.3. PyTorch is supplied by the hardware runtime; its actual build, Python version and device are recorded in notebook outputs.
3. The data cell rebuilds from local raw CNR assets if present, or downloads the documented prepared subset. Each experiment starts from the official pretrained model and uses the same split.
4. Default `RUN_TRAINING=True` trains all four settings, compares validation, evaluates the selected model, and displays all plots inline. Set `RUN_TRAINING=False` to skip new training and try the saved selected model, using the released metadata and weights.
5. The final cell saves selected weights and a compact archive of all four best checkpoints, configurations and epoch logs. Temporary training folders are removed only after the notebook outputs and checkpoint archive are produced. No separate visualization directory is retained.

For a new photo, use `selected_model.predict(source="your_image.jpg", imgsz=int(chosen["imgsz"]), conf=0.25, iou=0.7, save=False)`. Display `result.plot()` through Matplotlib. New-location accuracy is not established by this demonstration.

## Files and evidence

- `1.ipynb`: the single project notebook; all code, installation versions, data/model configuration, training histories, learning curves, five ground-truth examples, five predictions and three error cases are embedded.
- `README.md`: dataset, run instructions, comparison, interpretation and limitations.
- `parking_yolo11n.pt`: selected trained model.
- `Parking_Detection_Defense.pptx`: English presentation with speaker notes and real experiment evidence.
- `model_metadata.json`: selected setting, test metrics, split counts, evaluation settings and error audit.
- [Four best checkpoints and full training logs](https://github.com/Alihansagat/Deep-Learning/releases/download/parking-four-experiments-v1/experiment_checkpoints.zip): one ZIP containing each best checkpoint, `args.yaml`, `results.csv` and full run metadata.
- [Selected checkpoint download](https://github.com/Alihansagat/Deep-Learning/releases/download/parking-four-experiments-v1/parking_yolo11n.pt) and [metadata download](https://github.com/Alihansagat/Deep-Learning/releases/download/parking-four-experiments-v1/model_metadata.json).

## Limitations and practical next step

The same fixed cameras and parking lot appear across splits. Date grouping reduces same-day leakage but does not establish generalization to new cameras, lots or university conditions. Space boxes come from dataset geometry. Low-confidence extra detections may inflate availability counts. No live camera service, campus trial or real-time benchmark is claimed.

Compare results within this fixed 10-epoch budget; lower-LR runs may need more epochs. Only one seed is used. For campus use, collect local labeled images, evaluate a held-out new view, and select a confidence threshold on validation data that limits false free-space detections. Measure count error and errors by class before deployment.

## Assistance and individual defense

The project uses Ultralytics, scikit-learn, Matplotlib and other credited packages. Codex assisted with code, experiment orchestration and presentation preparation. The student should understand the data join, split, hyperparameters, metrics, model selection and three failure cases, and be able to make a small code change during the defense.

Submit the repository link to the instructor on Teams with your full name and student ID. The notebook does not send messages or include credentials. Student ID must be supplied by the student.

Baseline recovery: E0 was resumed after epoch 6 with optimizer state following an input-directory interruption. Logs were restored from the checkpoint itself. All runs share the same dataset snapshot and planned settings; exact bitwise reproducibility after interruption is not claimed.

### Observed failure explanations

**cam09_201511211717.jpg — Occupied spaces called free.** In low light, occupied spaces are labeled free; extra boxes also appear near trees. Next: Add labeled night examples and evaluate false free-space reports separately.

**cam09_201511211747.jpg — Missed spaces at night.** Very dark regions contain many labeled spaces that the detector misses or misaligns. Next: Improve camera exposure and evaluate a separate night subset.

**cam01_201601121217.jpg — ROI boundaries and label ambiguity.** Some extra boxes lie outside labeled slots. A free-labeled box overlaps the foreground van. Next: Review slot boundaries and ambiguous labels before validating ROI filtering.

Training times were measured on a shared local computer, not a controlled hardware speed benchmark.

Dataset YAML is regenerated with explicit train/val/test image lists from the manifest. Unrelated file-sync copies cannot silently enter training or evaluation. The recorded runs used a private snapshot with 965/288/347 images; validation and test logs confirm these counts.
