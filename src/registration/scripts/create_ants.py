import sys
import argparse
import os
from pathlib import Path
import zarr


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from registration.scripts.ants_classes import command_apply_other_resolution, command_register, configure_logging







class AntsRegistration:
    """
    This class is used to register images using ANTsPy
    """
    def __init__(self, moving, fixed, downsample, transformation, debug=False):
        self.moving = moving
        self.fixed = fixed
        self.downsample = downsample
        scratch_dir = "/data/pipeline_tmp"
        self.base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
        self.reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"
        self.moving_tif_path = os.path.join(self.base_path, self.moving, 'preps', 'C1', f'source_aligned.{self.downsample}')
        self.fixed_tif_path = os.path.join(self.base_path, self.fixed, 'preps', 'C1', f'source_aligned.{self.downsample}')
        self.registered_zarr_path = os.path.join(scratch_dir, self.moving, f'{self.moving}_{self.fixed}_registered.{self.downsample}.zarr')
        self.moving_zarr_path = os.path.join(scratch_dir, self.moving, f'source.{self.downsample}.zarr')
        self.fixed_zarr_path = os.path.join(scratch_dir, self.fixed, f'source.{self.downsample}.zarr')
        self.registered_tif_path = os.path.join(scratch_dir, self.moving, f'registered.{self.downsample}')
        self.xy_resolution = 0.325
        self.z_resolution = 20.0                
        self.transform_path = os.path.join(self.reg_path, f"{self.moving}_{self.fixed}.tfm")
        self.transformation = transformation
        self.debug = debug

    def status(self):
        brain_paths = [self.moving_zarr_path, self.fixed_zarr_path, self.registered_zarr_path]
        for brain_path in brain_paths:
            if os.path.exists(brain_path):
                zarr_data = zarr.open(brain_path, mode='r')
                print(brain_path)
                print(zarr_data.info)
            else:
                print(f'Missing: {brain_path}')
    def register_downsampled(self):
        #parser = build_parser()
        configure_logging()
        #args.func(args)
        command_register(self.moving, self.fixed, self.downsample, self.transformation, self.debug)

    def register_other_resolution(self):
        configure_logging()
        command_apply_other_resolution(self.moving, self.fixed, self.downsample)

    def get_info(self):
        scratch_dir = "/data/pipeline_tmp"
        output_zarr_path = os.path.join(scratch_dir, moving, f'{moving}_{fixed}_registered.{downsample}.zarr')
        moving_zarr_path = os.path.join(scratch_dir, moving, f'source.{downsample}.zarr')

        for p in [output_zarr_path, moving_zarr_path]:
            if os.path.exists(p):
                print(p)
                volume = zarr.open(p, mode='r')
                print(volume.info)
            else:
                print(f'Missing: {p}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument('--fixed', help='Enter the animal (fixed)', required=True, type=str)
    parser.add_argument('--downsample', help='Enter the downsample', required=True, type=int)
    parser.add_argument('--task', help='Enter the task', required=True, type=str)
    parser.add_argument('--transformation', help='Enter the transformation', required=False, default='Affine', type=str)
    parser.add_argument('--debug', help='Enter the debug', required=False, default='false', type=str)
    
    args = parser.parse_args()
    moving = args.moving
    fixed = args.fixed
    downsample = args.downsample
    task = str(args.task).strip().lower()
    transformation = str(args.transformation).strip()
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])

    pipeline = AntsRegistration(moving, fixed, downsample, transformation, debug=debug)


    function_mapping = {
        "register_downsampled": pipeline.register_downsampled,
        "register_other": pipeline.register_other_resolution,
        "get_info": pipeline.get_info
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
