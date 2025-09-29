import os
import shutil
import glob
import random
from pathlib import Path

def merge_deepotsu_to_existing_dataset(deepotsu_path, existing_dataset_path, train_ratio=0.8, val_ratio=0.1):
    """
    Merge deepotsu dataset into existing train/test/val structure
    
    Args:
        deepotsu_path: Path to deepotsu_dataset folder
        existing_dataset_path: Path to existing dataset (e.g., DataSet1KHDIB)
        train_ratio: Ratio of images to add to train set (default 0.8)
        val_ratio: Ratio of images to add to val set (default 0.1)
        # test_ratio is automatically 1 - train_ratio - val_ratio
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
        # Extract base name and replace 'in' with 'gt'
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
                    continue
            else:
                continue
        
        if os.path.exists(gt_file):
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
        
        # Copy files with sequential naming
        for i, (input_file, gt_file) in enumerate(pairs):
            # Generate new filenames
            file_extension = Path(input_file).suffix
            new_input_name = f"input_{start_count + i}{file_extension}"
            new_gt_name = f"mask_{start_count + i}{file_extension}"
            
            # Copy files
            input_dest = os.path.join(input_dir, new_input_name)
            gt_dest = os.path.join(gt_dir, new_gt_name)
            
            try:
                shutil.copy2(input_file, input_dest)
                shutil.copy2(gt_file, gt_dest)
            except Exception as e:
                print(f"Error copying {input_file}: {e}")
                continue
            
            if (i + 1) % 100 == 0:
                print(f"  Copied {i + 1}/{len(pairs)} pairs to {split_name}")
    
    print("Dataset merge completed successfully!")
    
    # Print final statistics
    for split_name in ['train', 'val', 'test']:
        input_dir = os.path.join(existing_dataset_path, f"Input_{split_name}")
        gt_dir = os.path.join(existing_dataset_path, f"GT_{split_name}")
        
        if os.path.exists(input_dir) and os.path.exists(gt_dir):
            input_count = len(glob.glob(os.path.join(input_dir, "*")))
            gt_count = len(glob.glob(os.path.join(gt_dir, "*")))
            print(f"{split_name.capitalize()} set: {input_count} inputs, {gt_count} ground truths")

def detect_image_pairs(directory_path):
    """
    Helper function to detect and list all input-ground truth pairs recursively
    """
    input_pattern = os.path.join(directory_path, "**", "*in.*")
    input_files = glob.glob(input_pattern, recursive=True)
    
    pairs = []
    for input_file in input_files:
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
                    continue
            else:
                continue
        
        if os.path.exists(gt_file):
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
    
    # Preview what will be merged
    pairs = detect_image_pairs(DEEPOTSU_DATASET_PATH)
    print(f"\nPreview: Found {len(pairs)} image pairs in deepotsu dataset (all subdirectories)")
    
    if len(pairs) > 0:
        print("\nSample pairs:")
        for i, (input_file, gt_file) in enumerate(pairs[:5]):  # Show first 5
            rel_input = os.path.relpath(input_file, DEEPOTSU_DATASET_PATH)
            rel_gt = os.path.relpath(gt_file, DEEPOTSU_DATASET_PATH)
            print(f"  {i+1}. Input: {rel_input} -> GT: {rel_gt}")
        
        # Ask for confirmation
        response = input(f"\nProceed to merge {len(pairs)} pairs into existing dataset? (y/n): ")
        
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