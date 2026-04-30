import argparse
import os
import sys
import numpy as np
import tifffile


def get_tif_files(directory):
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.tif', '.tiff'))
    ])


def compare_directories(dir1, dir2):
    files1 = get_tif_files(dir1)
    files2 = get_tif_files(dir2)
    all_ok = True

    if len(files1) == 0 or len(files2) == 0:
        print(f"[ERROR] One of the directories is empty: {dir1} ({len(files1)} files), {dir2} ({len(files2)} files)")
        return False

    # Check file counts
    if len(files1) != len(files2):
        print(f"[ERROR] File count mismatch: {len(files1)} vs {len(files2)}")
        return False

    # Check filenames
    if files1 != files2:
        print("[ERROR] Filenames do not match.")
        missing_in_dir2 = set(files1) - set(files2)
        missing_in_dir1 = set(files2) - set(files1)

        if missing_in_dir2:
            print(f"  Missing in dir2: {missing_in_dir2}")
        if missing_in_dir1:
            print(f"  Missing in dir1: {missing_in_dir1}")
        return False

    print(f"[INFO] {len(files1)} files matched by name.")


    for fname in files1:
        path1 = os.path.join(dir1, fname)
        path2 = os.path.join(dir2, fname)

        try:
            img1 = tifffile.imread(path1)
            img2 = tifffile.imread(path2)
        except Exception as e:
            print(f"[ERROR] Failed to read {fname}: {e}")
            all_ok = False
            continue

        # Check shape
        if img1.shape != img2.shape:
            print(f"[ERROR] Shape mismatch in {fname}: {img1.shape} vs {img2.shape}")
            all_ok = False
            continue

        # Optional pixel comparison

    if all_ok:
        print("[SUCCESS] All files match.")
    else:
        print("[DONE] Differences found.")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--c1", help="Enter the 1st dir", required=True, type=str)
    parser.add_argument("--c2", help="Enter the 2nd dir", required=True, type=str)
    args = parser.parse_args()

    animal = args.animal

    base_dir = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'
    dir1 = os.path.join(base_dir, args.c1)
    dir2 = os.path.join(base_dir, args.c2)

    compare_directories(dir1, dir2)