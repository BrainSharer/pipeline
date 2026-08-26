import sys
import argparse
import os
from pathlib import Path
import ants
import numpy as np
import pandas as pd


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.controller.sql_controller import SqlController


class AntsRegistration:
    """
    This class is used to register images using ANTsPy
    """
    def __init__(self, moving, fixed, downsample, transformation, debug=False):
        self.moving = moving
        self.fixed = fixed
        self.downsample = downsample
        self.base_path = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data"
        self.reg_path = "/net/birdstore/Active_Atlas_Data/data_root/brains_info/registration"

        moving_brain_controller = SqlController(self.moving)
        self.moving_xy_resolution = moving_brain_controller.scan_run.resolution
        self.moving_z_resolution = moving_brain_controller.scan_run.zresolution

        self.fixed_image_path = os.path.join(self.reg_path, self.fixed, f'source.{self.downsample}.nii')
        self.moving_image_path = os.path.join(self.reg_path, self.moving, f'source.{self.downsample}.nii')

        fixed_brain_controller = SqlController(self.fixed)
        self.fixed_xy_resolution = fixed_brain_controller.scan_run.resolution
        self.fixed_z_resolution = fixed_brain_controller.scan_run.zresolution

        self.moving_spacing = [ round(self.moving_xy_resolution,2), round(self.moving_xy_resolution,2), self.moving_z_resolution ]
        self.fixed_spacing = [ round(self.fixed_xy_resolution,2), round(self.fixed_xy_resolution,2), self.fixed_z_resolution ]

       
        self.transformation = transformation
        self.transform_path = os.path.join(self.reg_path, f"{self.moving}_{self.fixed}_{self.downsample}_{self.transformation}.mat")
        self.debug = debug

    def register_downsampled(self):
        # 1. Load the 3D brain volumes
        if os.path.exists(self.transform_path):
            #transform = ants.read_transform(self.transform_path)
            print(f"Loaded transform from: {self.transform_path}")
        else:
            if not os.path.exists(self.fixed_image_path):
                print(f'Missing: {self.fixed_image_path}')
                exit(0)
            if not os.path.exists(self.moving_image_path):
                print(f'Missing: {self.moving_image_path}')
                exit(0)

            fixed_img = ants.image_read(self.fixed_image_path)
            moving_img = ants.image_read(self.moving_image_path)
            print('Loaded fixed image', fixed_img)
            print('Loaded moving image', moving_img)
            print('Registering with', self.transformation)

            # 2. Run the Affine Registration
            # 'Affine' performs translation, rotation, scale, and shear
            reg = ants.registration(
                fixed=fixed_img,
                moving=moving_img,
                type_of_transform=self.transformation,
                verbose=self.debug
            )
            print(f'Finished registering')

            # Save the first transform path from the list
            transform_file = reg["fwdtransforms"][0]
            transform = ants.read_transform(transform_file)
            ants.write_transform(transform, self.transform_path)

        # 3. Define physical points in the moving brain space
        # Example: 3 coordinates in physical (RAS/LPS) space

        moving_points = np.array([
            [10317.416354106319, 3197.2081350069293, 5344.143279334614],
            [11970.730882949332, 5583.081898770125, 4611.1173576950105],
            [11977.032950062061, 5566.141666052378, 5313.9534883720935]
        ])

        # ANTsPy requires a pandas DataFrame with specific column names (x, y, z)
        moving_pts_df = pd.DataFrame(moving_points, columns=['x', 'y', 'z'])

        # 4. Transform points to the fixed brain space
        # Note: ANTs registration outputs moving-to-fixed transforms ('fwdtransforms')
        fixed_pts_df = ants.apply_transforms_to_points(
            dim=3,
            points=moving_pts_df,
            transformlist=[self.transform_path],
            whichtoinvert=[True])

        # Convert back to a numpy array
        fixed_points = fixed_pts_df.to_numpy()
        fixed_spacing = np.array(self.fixed_spacing)
        fixed_points = fixed_points / fixed_spacing

        print('Structures SC 6N_L 6N_R')
        print("Transformed points in fixed brain space:\n", fixed_points)        

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
    transformations = ['Affine', 'Rigid', 'SyN','ElasticSyN','SyNCC']
    if transformation not in transformations:
        print(f'{transformation} is not in {transformations}')
        exit(0)

    pipeline = AntsRegistration(moving, fixed, downsample, transformation, debug=debug)


    function_mapping = {
        "register": pipeline.register_downsampled,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
