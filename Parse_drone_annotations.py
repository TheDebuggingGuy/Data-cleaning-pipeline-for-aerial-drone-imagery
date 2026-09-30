"""
Drone Flight Log / Annotation Parser & Cleaner
------------------------------------------------
Parses YOLO-format bounding box annotations from a drone image dataset,
validates them, removes broken/malformed entries, and outputs:
  1. A clean, analysis-ready CSV
  2. Summary statistics (class distribution, image counts, etc.)
  3. Two charts: class distribution + bounding box size histogram

HOW TO USE WITH A REAL DATASET (e.g. VisDrone, tarlanahad/drone-detection-dataset):
  - Most drone/object-detection datasets ship annotations in one of two formats:
      a) YOLO .txt files: one per image, lines like "class x_center y_center width height" (normalized 0-1)
      b) VisDrone .txt files: "x,y,w,h,score,class,truncation,occlusion" (pixel coords, not normalized)
  - Point IMAGES_DIR and LABELS_DIR below at your downloaded dataset folders.
  - If your dataset uses the VisDrone format instead of YOLO, set FORMAT = "visdrone" below.

For now, this script runs in DEMO MODE: it generates a small synthetic
sample dataset (with some intentionally broken annotations) so you can
see the full pipeline working end-to-end without downloading anything.
"""

import os
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
IMAGES_DIR = "sample_data/images"
LABELS_DIR = "sample_data/labels"
FORMAT = "yolo"  # "yolo" or "visdrone"
CLASS_NAMES = ["pedestrian", "car", "bicycle", "van", "truck"]
OUTPUT_CSV = "clean_annotations.csv"
DEMO_MODE = True  # set to False once you point this at a real dataset


# ----------------------------
# STEP 0: Generate a small synthetic sample dataset (DEMO MODE only)
# ----------------------------
def generate_demo_data(n_images=25, seed=42):
    random.seed(seed)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    for i in range(n_images):
        img_name = f"img_{i:03d}.jpg"
        # create an empty placeholder "image" file (just a stand-in, not a real jpg)
        open(os.path.join(IMAGES_DIR, img_name), "a").close()

        label_path = os.path.join(LABELS_DIR, f"img_{i:03d}.txt")
        n_objects = random.randint(1, 6)
        lines = []
        for _ in range(n_objects):
            cls = random.randint(0, len(CLASS_NAMES) - 1)
            x_center = round(random.uniform(0, 1), 4)
            y_center = round(random.uniform(0, 1), 4)
            width = round(random.uniform(0.02, 0.3), 4)
            height = round(random.uniform(0.02, 0.3), 4)

            # intentionally corrupt ~15% of annotations to simulate real messy data
            if random.random() < 0.15:
                corruption = random.choice(["out_of_range", "missing_field", "bad_class"])
                if corruption == "out_of_range":
                    x_center = 1.4  # invalid, >1
                elif corruption == "missing_field":
                    lines.append(f"{cls} {x_center} {y_center} {width}")  # missing height
                    continue
                elif corruption == "bad_class":
                    cls = 99  # invalid class id

            lines.append(f"{cls} {x_center} {y_center} {width} {height}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    # also simulate a couple of images with NO matching label file (missing annotation)
    for i in range(n_images, n_images + 2):
        img_name = f"img_{i:03d}.jpg"
        open(os.path.join(IMAGES_DIR, img_name), "a").close()

    print(f"[demo] Generated {n_images + 2} sample images and annotations in '{IMAGES_DIR}' / '{LABELS_DIR}'")


# ----------------------------
# STEP 1: Parse raw annotation files
# ----------------------------
def parse_yolo_annotations(images_dir, labels_dir):
    records = []
    issues = []

    image_files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg")))

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_dir, base_name + ".txt")

        if not os.path.exists(label_file):
            issues.append({"image": img_file, "issue": "missing_label_file"})
            continue

        with open(label_file, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            issues.append({"image": img_file, "issue": "empty_label_file"})
            continue

        for line_num, line in enumerate(lines):
            parts = line.split()

            if len(parts) != 5:
                issues.append({"image": img_file, "issue": f"malformed_line_{line_num} (expected 5 fields, got {len(parts)})"})
                continue

            try:
                cls_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                issues.append({"image": img_file, "issue": f"non_numeric_value_line_{line_num}"})
                continue

            # validate class id
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                issues.append({"image": img_file, "issue": f"invalid_class_id_{cls_id}_line_{line_num}"})
                continue

            # validate normalized coordinates (must be within 0-1)
            coords = [x_center, y_center, width, height]
            if any(c < 0 or c > 1 for c in coords):
                issues.append({"image": img_file, "issue": f"out_of_range_bbox_line_{line_num}"})
                continue

            records.append({
                "image": img_file,
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "bbox_area": round(width * height, 5),
            })

    return pd.DataFrame(records), pd.DataFrame(issues)


# ----------------------------
# STEP 2: Summary stats + charts
# ----------------------------
def summarize_and_plot(df, issues_df):
    print("\n=== SUMMARY ===")
    print(f"Valid annotated objects: {len(df)}")
    print(f"Unique images with valid annotations: {df['image'].nunique()}")
    print(f"Issues found & discarded: {len(issues_df)}")

    if len(issues_df) > 0:
        print("\nIssue breakdown:")
        print(issues_df["issue"].apply(lambda x: x.split("_line_")[0] if "_line_" in x else x).value_counts())

    if df.empty:
        print("No valid data to plot.")
        return

    # Chart 1: class distribution
    plt.figure(figsize=(6, 4))
    df["class_name"].value_counts().plot(kind="bar", color="#4C72B0")
    plt.title("Class Distribution (valid annotations)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()

    # Chart 2: bounding box size histogram
    plt.figure(figsize=(6, 4))
    plt.hist(df["bbox_area"], bins=20, color="#DD8452")
    plt.title("Bounding Box Area Distribution (normalized)")
    plt.xlabel("Box area (fraction of image)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("bbox_area_distribution.png")
    plt.close()

    print("\nSaved charts: class_distribution.png, bbox_area_distribution.png")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    if DEMO_MODE:
        generate_demo_data()

    df, issues_df = parse_yolo_annotations(IMAGES_DIR, LABELS_DIR)

    df.to_csv(OUTPUT_CSV, index=False)
    issues_df.to_csv("discarded_issues.csv", index=False)
    print(f"\nClean data saved to '{OUTPUT_CSV}'")
    print(f"Discarded/invalid entries logged to 'discarded_issues.csv'")

    summarize_and_plot(df, issues_df)