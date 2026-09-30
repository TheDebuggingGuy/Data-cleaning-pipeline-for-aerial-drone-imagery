# Drone Annotation Data Cleaning Pipeline

A lightweight pipeline that parses, validates, and cleans object-detection
annotations from drone imagery (YOLO-format bounding boxes), then generates
summary statistics and visualizations.

## What it does
1. **Parses** per-image annotation files (class + bounding box coordinates)
2. **Validates** each entry — flags malformed lines, out-of-range coordinates,
   invalid class IDs, and images with missing/empty label files
3. **Cleans** the dataset — discards invalid entries into a separate log
   rather than silently dropping or crashing on them
4. **Exports** a clean, analysis-ready CSV (`clean_annotations.csv`) plus a
   log of everything discarded (`discarded_issues.csv`)
5. **Visualizes** class distribution and bounding box size distribution

## Why this matters
Real-world drone datasets are rarely clean — corrupted annotation files,
mislabeled classes, and missing files are common. This pipeline demonstrates
handling that messiness in a structured, auditable way rather than just
running everything through a model and hoping for the best.

## Running it
```bash
python3 parse_drone_annotations.py
```

By default it runs in **DEMO MODE**, generating a small synthetic sample
dataset (with intentionally corrupted entries) so you can see the full
pipeline work without downloading anything.

## Using it on a real dataset
1. Download a dataset such as:
   - [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
   - [tarlanahad/drone-detection-dataset](https://github.com/tarlanahad/drone-detection-dataset)
2. In `parse_drone_annotations.py`, set:
   ```python
   DEMO_MODE = False
   IMAGES_DIR = "path/to/your/images"
   LABELS_DIR = "path/to/your/labels"
   CLASS_NAMES = [...]  # match your dataset's classes
   ```
3. If your dataset uses VisDrone's raw annotation format instead of YOLO
   (`x,y,w,h,score,class,truncation,occlusion` in pixel coordinates), you'll
   need to adjust the `parse_yolo_annotations` function — a `parse_visdrone_annotations`
   variant is a natural next step.

## Output
- `clean_annotations.csv` — one row per valid bounding box
- `discarded_issues.csv` — every entry that failed validation, with a reason
- `class_distribution.png` — bar chart of object counts per class
- `bbox_area_distribution.png` — histogram of bounding box sizes

## Possible extensions
- Add VisDrone-format parsing support
- Convert clean annotations to COCO JSON format
- Feed cleaned data into a YOLOv8 training run
- Add image-level integrity checks (corrupt/unreadable files)
-