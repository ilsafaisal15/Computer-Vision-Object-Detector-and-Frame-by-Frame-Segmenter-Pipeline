# Computer Vision Object Detector and Frame-by-Frame Segmenter Pipeline

A real-time pipeline that extracts, classifies, tracks and counts target shapes in video, with per-frame inference metrics. Two pipelines are included: a **classical OpenCV pipeline** and a **YOLOv8n-seg** pipeline.


## Requirements

- Google Colab (a T4 GPU speeds up YOLO: Runtime > Change runtime type).
- OpenCV, numpy, pandas, matplotlib (preinstalled in Colab). The notebook installs `ultralytics` and downloads the `yolov8n-seg.pt` weights, so internet access is needed for the YOLO section.

## How to run

1. Upload `cv_pipeline_task4.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. Choose Runtime > Run all.
3. At the end `cv_pipeline_results.zip` downloads with the videos, CSVs and figures.

**Video source:** by default a synthetic 640 × 480, 150-frame video is generated (3 circles, 2 rectangles, 2 triangles bouncing on a noisy background) with ground truth, so accuracy can be measured. To use your own video set `USE_UPLOAD = True` in section 1 and re-run from there. YOLO is trained on everyday objects (people, cars and so on), so it will not detect the synthetic shapes; use real footage for a meaningful YOLO run.

## Classical pipeline

1. Gaussian filter (7 × 7)
2. Adaptive threshold (Gaussian-weighted, block size 51, C = 10) and morphological closing
3. Outer contour extraction
4. Pixel-bound filter: area between 600 and 20,000 px
5. Shape classification (triangle, rectangle, circle) from polygon approximation and circularity
6. Nearest-centroid tracking with persistent IDs
7. Overlay: contour, class label with ID, and a live count banner

Tunable values are in the `PARAMS` dictionary (`GAUSS_K`, `BLOCK`, `C`, `MIN_AREA`, `MAX_AREA`, `MORPH_K`).

## YOLO pipeline

Pre-trained `yolov8n-seg` (COCO classes) produces boxes and instance masks, the same count overlay, and per-frame latency from Ultralytics. **This section was not run for the report**, so no YOLO measurements are reported.

## Outputs (in `outputs/`)

| File | Content |
|---|---|
| `classical_annotated.mp4`, `yolo_annotated.mp4` | Annotated videos |
| `pipeline_stages.png` | Input, filtered, thresholded and annotated frame |
| `metrics_classical.csv`, `metrics_yolo.csv` | Per-frame latency, FPS, detections and stage timings |
| `summary_table.csv`, `metrics_overview.png` | Side-by-side summary and plots |
| `parameter_sensitivity.csv` | F1 and timing for 12 block-size and kernel settings (synthetic video only) |

## Key results (classical pipeline, synthetic video, CPU)

- Precision 0.938, recall 0.876, F1 0.906 (920 true positives, 61 false positives, 130 false negatives).
- Mean latency 4.27 ms per frame (about 234 FPS); adaptive thresholding takes about 61% of the time.
- Count error 0.46 objects per frame; 9 track IDs were created for 7 objects.
- F1 stays between 0.898 and 0.907 across the parameter sweep.

Timings vary between runs and machines.

## Limitations

- Synthetic, high-contrast video; accuracy on real footage will likely be lower.
- Misses are most likely overlapping shapes merging into one contour (not checked frame by frame).
- The centroid tracker has no motion model (no Kalman filter).
- No YOLO results yet: run section 5 on real footage to add them.
