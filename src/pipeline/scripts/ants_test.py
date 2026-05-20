import argparse
import os
import ants
import numpy as np
import tifffile
from tqdm import tqdm

def create_moving_image(base_path):

    file_path = os.path.join(base_path, 'C1', 'thumbnail_aligned')

    files = sorted(os.listdir(file_path))
    files = [os.path.join(file_path, f) for f in tqdm(files)]
    vol = np.stack([tifffile.imread(f) for f in tqdm(files)])
    spacing = (10.4, 10.4, 20.0)
    spacing_mm = tuple(s / 1000.0 for s in spacing)
    moving = ants.from_numpy(
        vol.astype(np.float32),
        spacing=spacing_mm
    )

    outpath = os.path.join(base_path, "moving.nii")
    ants.image_write(moving, outpath)
    print('Saved moving image to', outpath)
    # step 1
    moving = (moving - moving.min()) / (moving.max() - moving.min())
    outpath = os.path.join(base_path, "moving_norm.nii")
    ants.image_write(moving, outpath)
    print('Saved normalized moving image to', outpath)
    # step 2 n4 bias correction
    moving = ants.n4_bias_field_correction(
        moving,
        shrink_factor=4
    )
    outpath = os.path.join(base_path, "moving_n4.nii")
    ants.image_write(moving, outpath)
    print('Saved N4 bias corrected moving image to', outpath)
    # step 3 denoising
    moving = ants.denoise_image(
        moving,
        noise_model='Gaussian'
    )
    outpath = os.path.join(base_path, "moving_denoised.nii")
    ants.image_write(moving, outpath)
    print('Saved denoised moving image to', outpath)
    # step 4 mask
    mask = ants.get_mask(
        moving,
        low_thresh=0.05,
        high_thresh=1.0
    )
    mask = ants.iMath(mask, "FillHoles")
    mask = ants.iMath(mask, "ME", 2)
    mask = ants.iMath(mask, "MD", 2)
    outpath = os.path.join(base_path, "moving_mask.nii")
    ants.image_write(mask, outpath)
    print('Saved mask to', outpath)
    moving_iso = ants.resample_image(
        moving,
        resample_params=(0.025, 0.025, 0.025),
        use_voxels=False,
        interp_type=1
    )
    outpath = os.path.join(base_path, "moving_iso.nii")
    ants.image_write(moving_iso, outpath)
    print('Saved isotropic moving image to', outpath)

    mask_iso = ants.resample_image_to_target(
        mask,
        moving_iso,
        interp_type='nearestNeighbor'
    )
    outpath = os.path.join(base_path, "moving_mask_iso.nii")
    ants.image_write(mask_iso, outpath)
    print('Saved isotropic mask to', outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    args = parser.parse_args()

    animal = args.animal

    base_dir = f'/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps'


    create_moving_image(base_dir)
