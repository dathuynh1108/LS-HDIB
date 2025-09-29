import os
import shutil
import glob
import random
import cv2
import numpy as np
from pathlib import Path

# ------------------------------ Core helpers ------------------------------ #


def invert_ground_truth(gt_image_path):
    """
    Invert ground-truth image to match training format
    DeepOTSU: black objects on white background
    Training: white objects on black background (THRESH_BINARY_INV format)
    Always returns a single-channel (grayscale) image.
    """
    gt_img = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        return None
    return 255 - gt_img


def _find_corresponding_gt_file(input_file):
    """
    Find corresponding ground truth file for given input file.
    DeepOTSU naming is typically ...in.{ext} and ...gt.{ext}.
    Handles both '.in.' and 'in' without dot.
    """
    if "in." in input_file:
        base_path = input_file.rsplit("in.", 1)[0]
        extension = input_file.rsplit(".", 1)[1]
        gt_file = f"{base_path}gt.{extension}"
        return gt_file

    # Handle case where 'in' is part of filename without dot
    parts = input_file.split("in")
    if len(parts) >= 2:
        base_path = "in".join(parts[:-1])
        extension_part = parts[-1]
        if "." in extension_part:
            extension = extension_part.split(".", 1)[1]
            gt_file = f"{base_path}gt.{extension}"
            return gt_file

    return None


def detect_image_pairs(directory_path):
    """
    Recursively detect and list all input-ground truth pairs in a DeepOTSU folder.
    Returns: list[(input_path, gt_path)]
    """
    input_pattern = os.path.join(directory_path, "**", "*in.*")
    input_files = glob.glob(input_pattern, recursive=True)
    pairs = []
    for input_file in input_files:
        gt_file = _find_corresponding_gt_file(input_file)
        if gt_file and os.path.exists(gt_file):
            pairs.append((input_file, gt_file))
    return pairs


def preview_dataset_structure(directory_path):
    """
    Preview the structure of the deepotsu dataset
    """
    print(f"\nDataset structure preview for: {directory_path}")
    print("-" * 50)

    subdirs = []
    for root, dirs, files in os.walk(directory_path):
        if root != directory_path:
            rel_path = os.path.relpath(root, directory_path)
            subdirs.append(rel_path)

    if subdirs:
        print(f"Found {len(subdirs)} subdirectories:")
        for subdir in sorted(subdirs)[:10]:  # show first 10
            input_files = glob.glob(os.path.join(directory_path, subdir, "*in.*"))
            gt_files = glob.glob(os.path.join(directory_path, subdir, "*gt.*"))
            print(
                f"  {subdir}: {len(input_files)} input files, {len(gt_files)} gt files"
            )
        if len(subdirs) > 10:
            print(f"  ... and {len(subdirs) - 10} more subdirectories")
    else:
        print("No subdirectories found - all files are in root directory")


def preview_image_formats(directory_path, max_samples=3):
    """
    Preview input and ground-truth image formats to verify inversion is needed.
    """
    print(f"\nImage format preview:")
    print("-" * 30)

    pairs = detect_image_pairs(directory_path)
    if not pairs:
        print("No image pairs found for preview")
        return

    for i, (input_file, gt_file) in enumerate(pairs[:max_samples]):
        print(f"\nSample {i+1}:")
        print(f"  Input: {os.path.relpath(input_file, directory_path)}")
        print(f"  GT:    {os.path.relpath(gt_file, directory_path)}")

        gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        if gt_img is not None:
            unique_values = np.unique(gt_img)
            mean_value = float(np.mean(gt_img))
            fmt = (
                "Black objects on white background"
                if mean_value > 127
                else "White objects on black background"
            )
            print(
                f"  GT unique values: {unique_values[:10]}{'...' if len(unique_values)>10 else ''}"
            )
            print(f"  GT mean value: {mean_value:.1f}")
            print(f"  GT format: {fmt}")
        else:
            print("  GT could not be read.")


# ------------------------------ Merge function ------------------------------ #


def merge_deepotsu_to_existing_dataset(
    deepotsu_path,
    existing_dataset_path,
    train_ratio=0.8,
    val_ratio=0.1,
    jpeg_quality=100,
):
    """
    Merge DeepOTSU dataset into existing train/val/test structure with ground-truth inversion
    and normalize ALL saved files to .jpg.

    Args:
        deepotsu_path:         Path to deepotsu_dataset folder
        existing_dataset_path: Path to existing dataset (e.g., DataSet1KHDIB)
        train_ratio:           Ratio of images to add to train set (default 0.8)
        val_ratio:             Ratio of images to add to val set (default 0.1)
        jpeg_quality:          JPEG quality (0..100)
    """

    # Reproducibility
    random.seed(43)

    # Discover pairs
    input_pattern = os.path.join(deepotsu_path, "**", "*in.*")
    input_files = glob.glob(input_pattern, recursive=True)
    print(
        f"Found {len(input_files)} input images in deepotsu dataset (including subdirectories)"
    )

    valid_pairs = []
    for input_file in input_files:
        gt_file = _find_corresponding_gt_file(input_file)
        if gt_file and os.path.exists(gt_file):
            valid_pairs.append((input_file, gt_file))
        else:
            print(f"Warning: No ground truth found for {input_file}")

    print(f"Found {len(valid_pairs)} valid input-ground truth pairs")
    if not valid_pairs:
        print("No valid pairs found. Exiting.")
        return

    # Shuffle and split
    random.shuffle(valid_pairs)
    total_pairs = len(valid_pairs)
    train_end = int(total_pairs * train_ratio)
    val_end = train_end + int(total_pairs * val_ratio)

    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]

    print(
        f"Distribution: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}"
    )

    # Prepare output dirs
    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    for split_name, pairs in splits.items():
        if not pairs:
            continue

        input_dir = os.path.join(existing_dataset_path, f"Input_{split_name}")
        gt_dir = os.path.join(existing_dataset_path, f"GT_{split_name}")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        # Continue sequential numbering after existing files (robust to any extensions)
        existing_input_files = glob.glob(os.path.join(input_dir, "*"))
        existing_gt_files = glob.glob(os.path.join(gt_dir, "*"))
        start_count = max(len(existing_input_files), len(existing_gt_files)) + 1

        print(
            f"Adding {len(pairs)} pairs to {split_name} split (starting from index {start_count})"
        )

        processed_count = 0
        for _, (input_file, gt_file) in enumerate(pairs):
            try:
                # Filenames: always .jpg
                idx = start_count + processed_count
                new_input_name = f"input_{idx}.jpg"
                new_gt_name = f"mask_{idx}.jpg"

                input_dest = os.path.join(input_dir, new_input_name)
                gt_dest = os.path.join(gt_dir, new_gt_name)

                # ---- Read & Write INPUT as .jpg ----
                img = cv2.imread(input_file, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Warning: Could not read input {input_file}")
                    continue
                ok_input = cv2.imwrite(
                    input_dest, img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                )
                if not ok_input:
                    print(f"Warning: Could not write input JPG {input_dest}")
                    continue

                # ---- Invert & Write GT as .jpg ----
                inverted_gt = invert_ground_truth(gt_file)
                if inverted_gt is None:
                    print(f"Warning: Could not process ground truth {gt_file}")
                    # cleanup partially written input
                    if os.path.exists(input_dest):
                        os.remove(input_dest)
                    continue

                ok_gt = cv2.imwrite(
                    gt_dest,
                    inverted_gt,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not ok_gt:
                    print(f"Warning: Could not write GT JPG {gt_dest}")
                    if os.path.exists(input_dest):
                        os.remove(input_dest)
                    continue

                processed_count += 1
                if processed_count % 100 == 0:
                    print(
                        f"  Processed {processed_count}/{len(pairs)} pairs for {split_name}"
                    )

            except Exception as e:
                print(f"Error processing pair ({input_file}) -> ({gt_file}): {e}")
                # Attempt cleanup if needed
                try:
                    if os.path.exists(input_dest):
                        os.remove(input_dest)
                    if os.path.exists(gt_dest):
                        os.remove(gt_dest)
                except Exception:
                    pass
                continue

        print(f"Successfully processed {processed_count} pairs for {split_name} split")

    print("Dataset merge with ground truth inversion completed successfully!")

    # Final statistics
    for split_name in ["train", "val", "test"]:
        input_dir = os.path.join(existing_dataset_path, f"Input_{split_name}")
        gt_dir = os.path.join(existing_dataset_path, f"GT_{split_name}")

        if os.path.exists(input_dir) and os.path.exists(gt_dir):
            input_count = len(glob.glob(os.path.join(input_dir, "*")))
            gt_count = len(glob.glob(os.path.join(gt_dir, "*")))
            print(
                f"{split_name.capitalize()} set: {input_count} inputs, {gt_count} ground truths"
            )


# ------------------------------- CLI runner ------------------------------- #

if __name__ == "__main__":
    # Configuration
    DEEPOTSU_DATASET_PATH = "deepotsu_dataset"  # Path to your DeepOTSU dataset
    EXISTING_DATASET_PATH = "DataSet1KHDIB"  # Path to your target dataset root

    # Verify paths exist
    if not os.path.exists(DEEPOTSU_DATASET_PATH):
        print(f"Error: {DEEPOTSU_DATASET_PATH} does not exist")
        raise SystemExit(1)

    if not os.path.exists(EXISTING_DATASET_PATH):
        print(f"Error: {EXISTING_DATASET_PATH} does not exist")
        raise SystemExit(1)

    # Preview dataset structure & a few samples
    preview_dataset_structure(DEEPOTSU_DATASET_PATH)
    preview_image_formats(DEEPOTSU_DATASET_PATH, max_samples=3)

    # Preview how many pairs exist
    pairs = detect_image_pairs(DEEPOTSU_DATASET_PATH)
    print(
        f"\nPreview: Found {len(pairs)} image pairs in deepotsu dataset (all subdirectories)"
    )
    if pairs:
        print("\nSample pairs:")
        for i, (input_file, gt_file) in enumerate(pairs[:5]):  # show first 5
            rel_input = os.path.relpath(input_file, DEEPOTSU_DATASET_PATH)
            rel_gt = os.path.relpath(gt_file, DEEPOTSU_DATASET_PATH)
            print(f"  {i+1}. Input: {rel_input} -> GT: {rel_gt}")

        print("\nNOTE: All outputs will be written as .jpg")
        print("      GT images will be inverted (white text on black background).")

        # Ask for confirmation
        resp = (
            input(
                f"\nProceed to merge {len(pairs)} pairs with JPG conversion and GT inversion? (y/n): "
            )
            .strip()
            .lower()
        )
        if resp == "y":
            merge_deepotsu_to_existing_dataset(
                DEEPOTSU_DATASET_PATH,
                EXISTING_DATASET_PATH,
                train_ratio=0.8,
                val_ratio=0.1,
                jpeg_quality=100,
            )
        else:
            print("Merge cancelled.")
    else:
        print("No valid image pairs found. Please check your dataset structure.")
