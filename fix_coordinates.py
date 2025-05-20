import os
import glob
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fix_label_file(label_path, max_coord=1.0, min_coord=0.0):
    """
    Fix a YOLO label file by clamping coordinates to valid ranges.
    
    Args:
        label_path: Path to the label file
        max_coord: Maximum allowed coordinate value (default 1.0)
        min_coord: Minimum allowed coordinate value (default 0.0)
    
    Returns:
        bool: True if the file was fixed, False if no fixing was needed
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        file_needed_fixing = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # Standard YOLO format has at least 5 parts
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:5]]
                
                # Check if any coordinates are out of bounds
                if any(x > max_coord or x < min_coord for x in coords):
                    file_needed_fixing = True
                    # Clamp coordinates to valid range
                    coords = [max(min(x, max_coord), min_coord) for x in coords]
                
                # Rebuild the line
                fixed_line = f"{class_id} {' '.join(f'{x:.6f}' for x in coords)}\n"
                fixed_lines.append(fixed_line)
            else:
                # Keep the line as is if it doesn't have enough parts
                fixed_lines.append(line)
        
        # Write the fixed lines back if any changes were made
        if file_needed_fixing:
            with open(label_path, 'w') as f:
                f.writelines(fixed_lines)
            
        return file_needed_fixing
    
    except Exception as e:
        logging.error(f"Error processing {label_path}: {e}")
        return False

def find_and_process_labels(data_dir):
    """
    Find and process label files in a directory structure
    
    Args:
        data_dir: Base directory for dataset (train or val)
    """
    # Try different possible locations for labels
    possible_label_dirs = [
        os.path.join(data_dir, "labels"),  # dataset/train/labels/
        data_dir,                          # dataset/train/ (labels directly in this dir)
    ]
    
    # Also check for any .txt files in the images directory structure
    images_dir = os.path.join(data_dir, "images")
    if os.path.exists(images_dir):
        possible_label_dirs.append(images_dir)  # dataset/train/images/ (sometimes labels are with images)
    
    labels_found = False
    
    # Try each possible location
    for label_dir in possible_label_dirs:
        if not os.path.exists(label_dir):
            continue
            
        # Look for .txt files in this directory
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        
        # Skip cache files
        label_files = [f for f in label_files if not os.path.basename(f) == "labels.cache"]
        
        if label_files:
            logging.info(f"Found {len(label_files)} label files in {label_dir}")
            
            fixed_count = 0
            for label_file in label_files:
                if fix_label_file(label_file):
                    fixed_count += 1
            
            logging.info(f"Fixed {fixed_count} out of {len(label_files)} label files")
            
            # If we fixed any files, delete the cache so it will be regenerated
            if fixed_count > 0:
                cache_file = os.path.join(label_dir, "labels.cache")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logging.info(f"Deleted cache file {cache_file}")
                    
                # Also check for cache in the main train directory
                cache_file = os.path.join(data_dir, "labels.cache")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logging.info(f"Deleted cache file {cache_file}")
            
            labels_found = True
    
    if not labels_found:
        logging.error(f"No label files found in or under {data_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix YOLO label files with out-of-bounds coordinates')
    parser.add_argument('--train_dir', type=str, required=True, 
                        help='Path to train directory (e.g., dataset/train)')
    parser.add_argument('--val_dir', type=str, required=False,
                        help='Path to validation directory (optional)')
    
    args = parser.parse_args()
    
    # Process training labels
    logging.info(f"Processing training dataset in {args.train_dir}")
    find_and_process_labels(args.train_dir)
    
    # Process validation labels if provided
    if args.val_dir:
        logging.info(f"Processing validation dataset in {args.val_dir}")
        find_and_process_labels(args.val_dir)
    
    logging.info("Done!")