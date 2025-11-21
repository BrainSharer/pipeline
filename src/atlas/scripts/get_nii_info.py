import argparse
import os
import numpy as np

import numpy as np
import SimpleITK as sitk


def get_nii_info(nii_path: str) -> None:
    """
    Load a NIfTI file and print its shape, spacing, origin, and direction.
    """
    if not os.path.exists(nii_path):
        print(f"NIfTI file not found: {nii_path}")
        return

    img = sitk.ReadImage(nii_path)
    #arr = sitk.GetArrayFromImage(img)

    print(f"NIfTI file: {nii_path}")
    #print(f"Shape (z,y,x): {arr.shape}")
    #print(f"Data type: {arr.dtype}")
    print(f"Spacing: {img.GetSpacing()}")
    print(f"Origin: {img.GetOrigin()}")
    print(f"Direction: {img.GetDirection()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get NIfTI file information')
    parser.add_argument('--nii_path', help='Path to the NIfTI file', required=True, type=str)
    args = parser.parse_args()
    get_nii_info(args.nii_path)