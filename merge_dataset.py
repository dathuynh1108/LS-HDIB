import os
import shutil
import glob
import random
import cv2
import numpy as np
from pathlib import Path

def invert_ground_truth(gt_image_path):
    """
    Invert ground truth image to match training format
    DeepOTSU: black objects on white background
    Training: white objects on black background (THRESH_BINARY_INV format)
    """
    gt_img = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        return None
    
    # Invert: 255 - pixel_value (white becomes black, black becomes white)
    inverted_gt = 255 - gt_img
    
    return inverted_gt

def merge_deepotsu_to_existing_dataset(deepotsu_path, existing_dataset_path, train_ratio=0.8, val_ratio=0.1):
    """
    Merge deepotsu dataset into existing train/test/val structure with ground truth inversion
    
    Args:
        deepotsu_path: Path to deepotsu_dataset folder
        existing_dataset_path: Path to existing dataset (e.g., DataSet1KHDIB)
        train_ratio: Ratio of images to add to train set (default 0.8)
        val_ratio: Ratio of images to add to val set (default 0.1)
    """
    
    # Set random seed for reproducibility
    random.seed(43)
    
    # Recursively find all input images (with 'in' postfix) in all subdirectories
    input_pattern = os.path.join(deepotsu_path, "**", "*in.*")
    input_files = glob.glob(input_pattern, recursive=True)
    
    print(f"Found {len(input_files)} input images in deepotsu dataset (including subdirectories)")
    
    # Group by subdirectory for better organization
    subdirs = {}
    for input_file in input_files:
        relative_path = os.path.relpath(input_file, deepotsu_path)
        subdir = os.path.dirname(relative_path)
        if subdir not in subdirs:
            subdirs[subdir] = []
        subdirs[subdir].append(input_file)
    
    print(f"Files distributed across {len(subdirs)} subdirectories:")
    for subdir, files in subdirs.items():
        print(f"  {subdir if subdir else '(root)'}: {len(files)} files")
    
    # Verify corresponding ground truth files exist
    valid_pairs = []
    for input_file in input_files:
        gt_file = _find_corresponding_gt_file(input_file)
        
        if gt_file and os.path.exists(gt_file):
            valid_pairs.append((input_file, gt_file))
        else:
            print(f"Warning: No ground truth found for {input_file}")
    
    print(f"Found {len(valid_pairs)} valid input-ground truth pairs")
    
    if len(valid_pairs) == 0:
        print("No valid pairs found. Exiting.")
        return
    
    # Shuffle for random distribution
    random.shuffle(valid_pairs)
    
    # Calculate split indices
    total_pairs = len(valid_pairs)
    train_end = int(total_pairs * train_ratio)
    val_end = train_end + int(total_pairs * val_ratio)
    
    # Split the data
    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]
    
    print(f"Distribution: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
    
    # Create target directories if they don't exist
    splits = {
        'train': train_pairs,
        'val': val_pairs, 
        'test': test_pairs
    }
    
    for split_name, pairs in splits.items():
        if len(pairs) == 0:
            continue
            
        # Create directories
        input_dir = os.path.join(existing_dataset_path, f"Input_{split_name}")
        gt_dir = os.path.join(existing_dataset_path, f"GT_{split_name}")
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        
        # Get current file count to avoid naming conflicts
        existing_input_files = len(glob.glob(os.path.join(input_dir, "*")))
        existing_gt_files = len(glob.glob(os.path.join(gt_dir, "*")))
        start_count = max(existing_input_files, existing_gt_files) + 1
        
        print(f"Adding {len(pairs)} pairs to {split_name} split (starting from index {start_count})")
        
        # Process and copy files with sequential naming
        processed_count = 0
        for i, (input_file, gt_file) in enumerate(pairs):
            try:
                # Generate new filenames
                file_extension = Path(input_file).suffix
                new_input_name = f"input_{start_count + processed_count}{file_extension}"
                new_gt_name = f"mask_{start_count + processed_count}{file_extension}"
                
                # Copy input file directly
                input_dest = os.path.join(input_dir, new_input_name)
                shutil.copy2(input_file, input_dest)
                
                # Process and save inverted ground truth
                gt_dest = os.path.join(gt_dir, new_gt_name)
                inverted_gt = invert_ground_truth(gt_file)
                
                if inverted_gt is not None:
                    cv2.imwrite(gt_dest, inverted_gt)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        print(f"  Processed {processed_count}/{len(pairs)} pairs for {split_name}")
                else:
                    print(f"Warning: Could not process ground truth {gt_file}")
                    # Remove the copied input file if GT processing failed
                    if os.path.exists(input_dest):
                        os.remove(input_dest)
                        
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                continue
        
        print(f"Successfully processed {processed_count} pairs for {split_name} split")
    
    print("Dataset merge with ground truth inversion completed successfully!")
    
    # Print final statistics
    for split_name in ['train', 'val', 'test']:
        input_dir = os.path.join(existing_dataset_path, f"Input_{split_name}")
        gt_dir = os.path.join(existing_dataset_path, f"GT_{split_name}")
        
        if os.path.exists(input_dir) and os.path.exists(gt_dir):
            input_count = len(glob.glob(os.path.join(input_dir, "*")))
            gt_count = len(glob.glob(os.path.join(gt_dir, "*")))
            print(f"{split_name.capitalize()} set: {input_count} inputs, {gt_count} ground truths")

def _find_corresponding_gt_file(input_file):
    """Helper function to find corresponding ground truth file"""
    if 'in.' in input_file:
        base_path = input_file.rsplit('in.', 1)[0]
        extension = input_file.rsplit('.', 1)[1]
        gt_file = f"{base_path}gt.{extension}"
    else:
        # Handle case where 'in' is part of filename without dot
        parts = input_file.split('in')
        if len(parts) >= 2:
            base_path = 'in'.join(parts[:-1])
            extension_part = parts[-1]
            if '.' in extension_part:
                extension = extension_part.split('.', 1)[1]
                gt_file = f"{base_path}gt.{extension}"
            else:
                return None
        else:
            return None
    
    return gt_file

def detect_image_pairs(directory_path):
    """
    Helper function to detect and list all input-ground truth pairs recursively
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
    
    # Find all subdirectories
    subdirs = []
    for root, dirs, files in os.walk(directory_path):
        if root != directory_path:  # Skip root directory
            rel_path = os.path.relpath(root, directory_path)
            subdirs.append(rel_path)
    
    if subdirs:
        print(f"Found {len(subdirs)} subdirectories:")
        for subdir in sorted(subdirs)[:10]:  # Show first 10
            input_files = glob.glob(os.path.join(directory_path, subdir, "*in.*"))
            gt_files = glob.glob(os.path.join(directory_path, subdir, "*gt.*"))
            print(f"  {subdir}: {len(input_files)} input files, {len(gt_files)} gt files")
        
        if len(subdirs) > 10:
            print(f"  ... and {len(subdirs) - 10} more subdirectories")
    else:
        print("No subdirectories found - all files are in root directory")

def preview_image_formats(directory_path, max_samples=3):
    """
    Preview input and ground truth image formats to verify inversion is needed
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
        print(f"  GT: {os.path.relpath(gt_file, directory_path)}")
        
        # Analyze ground truth image
        gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        if gt_img is not None:
            unique_values = np.unique(gt_img)
            mean_value = np.mean(gt_img)
            print(f"  GT unique values: {unique_values}")
            print(f"  GT mean value: {mean_value:.1f}")
            print(f"  GT format: {'Black objects on white background' if mean_value > 127 else 'White objects on black background'}")

if __name__ == "__main__":
    # Configuration
    DEEPOTSU_DATASET_PATH = "deepotsu_dataset"  # Path to your deepotsu dataset
    EXISTING_DATASET_PATH = "DataSet1KHDIB"    # Path to existing dataset
    
    # Verify paths exist
    if not os.path.exists(DEEPOTSU_DATASET_PATH):
        print(f"Error: {DEEPOTSU_DATASET_PATH} does not exist")
        exit(1)
    
    if not os.path.exists(EXISTING_DATASET_PATH):
        print(f"Error: {EXISTING_DATASET_PATH} does not exist")
        exit(1)
    
    # Preview dataset structure
    preview_dataset_structure(DEEPOTSU_DATASET_PATH)
    
    # Preview image formats to confirm inversion is needed
    preview_image_formats(DEEPOTSU_DATASET_PATH)
    
    # Preview what will be merged
    pairs = detect_image_pairs(DEEPOTSU_DATASET_PATH)
    print(f"\nPreview: Found {len(pairs)} image pairs in deepotsu dataset (all subdirectories)")
    
    if len(pairs) > 0:
        print("\nSample pairs:")
        for i, (input_file, gt_file) in enumerate(pairs[:5]):  # Show first 5
            rel_input = os.path.relpath(input_file, DEEPOTSU_DATASET_PATH)
            rel_gt = os.path.relpath(gt_file, DEEPOTSU_DATASET_PATH)
            print(f"  {i+1}. Input: {rel_input} -> GT: {rel_gt}")
        
        print("\nNOTE: Ground truth images will be inverted during merge process")
        print("DeepOTSU format: Black objects on white background")  
        print("Training format: White objects on black background")
        
        # Ask for confirmation
        response = input(f"\nProceed to merge {len(pairs)} pairs with GT inversion? (y/n): ")
        
        if response.lower() == 'y':
            merge_deepotsu_to_existing_dataset(
                DEEPOTSU_DATASET_PATH, 
                EXISTING_DATASET_PATH,
                train_ratio=0.8,
                val_ratio=0.1
            )
        else:
            print("Merge cancelled.")
    else:
        print("No valid image pairs found. Please check your dataset structure.")