import sys
import argparse
import os
from pathlib import Path
import ants
import numpy as np
import pandas as pd
import shutil
import glob
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from registration.scripts.sitk_helpers import get_points_from_db
from library.controller.sql_controller import SqlController
from library.registration.algorithm import umeyama
from library.utilities.utilities_process import M_UM_SCALE



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

    def testing(self):
        sqlController = SqlController(self.moving)
        moving_coms = sqlController.get_com_dictionary(self.moving)
        fixed_coms = sqlController.get_com_dictionary(self.fixed)
        common_landmarks = (set(moving_coms.keys()).intersection(set(fixed_coms.keys())))
        moving_coms = np.array([moving_coms[landmark] for landmark in common_landmarks])
        fixed_coms = np.array([fixed_coms[landmark] for landmark in common_landmarks])
        moving_points = moving_coms * M_UM_SCALE
        fixed_points = fixed_coms * M_UM_SCALE
        structures = common_landmarks

        for structure, f, m in zip(common_landmarks, fixed_points, moving_points):
            print(structure, np.round(f), np.round(m))


        A, t = umeyama(moving_points.T, fixed_points.T, with_scaling=False)
        transformation_matrix = np.hstack( [A, t ])
        transformation_matrix = np.vstack([transformation_matrix, np.array([0, 0, 0, 1])])

        # Step 1: Add a column of ones to make points Nx4
        ones = np.ones((moving_points.shape[0], 1))
        points_homogeneous = np.hstack([moving_points, ones])
        # Step 2: Apply the 4x4 matrix using dot product (Nx4 @ 4x4 -> Nx4)
        transformed_homogeneous = points_homogeneous @ transformation_matrix.T

        # Step 3: Extract the first 3 columns back to Nx3
        transformed_points = np.round(transformed_homogeneous[:, :3])

        print(f'Transformed points in {self.fixed} space')
        for structure, point, f in zip(structures, transformed_points, fixed_points):
            print(structure, np.round(point/self.fixed_spacing), np.linalg.norm(f-point))
        #print(transformed_points/fixed_spacing)
        distances = np.linalg.norm(fixed_points-transformed_points, axis=1)
        print('Distances using umeyama:')
        mean = float(np.mean(distances))
        median = float(np.median(distances))
        maximum = float(np.max(distances))
        print(
            f"\tmean={mean:.3f} µm, "
            f"median={median:.3f} µm, "
            f"max={maximum:.3f} µm"
        )



    def register_points(self):
        transforms = glob.glob(os.path.join(self.reg_path, f'{self.moving}_{self.fixed}_{self.transformation}_{self.downsample}_*'))
        transforms = sorted(transforms)
        # 1. Load the 3D brain volumes
        if len(transforms) > 0:
            print(f"Loaded transforms \n: {transforms}")
        else:
            print(f'There are no transforms for {self.moving}_{self.fixed}_{self.transformation}_{self.downsample}')
            exit(0)

        structures = ['10N_L','10N_R','3N_L','3N_R','4N_L','4N_R','5N_L','5N_R','6N_L','6N_R','7N_L','7N_R','7n_L', 'SC']
        fixed_points = get_points_from_db(self.fixed)
        moving_points = get_points_from_db(self.moving)



        # ANTsPy requires a pandas DataFrame with specific column names (x, y, z)
        moving_pts_df = pd.DataFrame(moving_points, columns=['x', 'y', 'z'])

        # 4. Transform points to the fixed brain space
        # Note: ANTs registration outputs moving-to-fixed transforms ('fwdtransforms')
        df = ants.apply_transforms_to_points(
            dim=3,
            points=moving_pts_df,
            transformlist=transforms,
            whichtoinvert=[True])

        # Convert back to a numpy array
        registered_points = df.to_numpy()

        fixed_spacing = np.array(self.fixed_spacing)
        registered_points = registered_points
        # fix up DF
        df['Structures'] = structures
        #df['x'] = df['x'] / fixed_spacing[0]
        #df['y'] = df['y'] / fixed_spacing[1]
        #df['z'] = df['z'] / fixed_spacing[2]
        df['Fixed'] = [self.fixed] * len(structures)
        df[["xf", "yf", "zf"]] = fixed_points
        #df['xf'] = df['xf'] / fixed_spacing[0]
        #df['yf'] = df['yf'] / fixed_spacing[1]
        #df['zf'] = df['zf'] / fixed_spacing[2]
        df = df.round(0)

        df['x'] = df['x'].astype(int)
        df['y'] = df['y'].astype(int)
        df['z'] = df['z'].astype(int)
        df['xf'] = df['xf'].astype(int)
        df['yf'] = df['yf'].astype(int)
        df['zf'] = df['zf'].astype(int)
        df = df.set_index('Structures')

        target_idx = df.columns.get_loc('x') + 1
        column_data = [self.moving] * len(structures)
        df.insert(loc=target_idx, column='Moving', value=column_data)
        col = df.pop('Moving')
        # 2. Insert it back at position 0 (right after the index)
        df.insert(0, 'Moving', col)
        distances = np.linalg.norm(fixed_points-registered_points, axis=1)
        df['Distances'] = distances

        print(df.head(20))
        mean = float(np.mean(distances))
        median = float(np.median(distances))
        maximum = float(np.max(distances))

        print(f'Distances using {self.transformation} transformation:')
        print(
            f"\tmean={mean:.3f} µm, "
            f"median={median:.3f} µm, "
            f"max={maximum:.3f} µm"
        )
        print()

        A, t = umeyama(moving_points.T, fixed_points.T, with_scaling=False)
        transformation_matrix = np.hstack( [A, t ])
        transformation_matrix = np.vstack([transformation_matrix, np.array([0, 0, 0, 1])])

        # Step 1: Add a column of ones to make points Nx4
        ones = np.ones((moving_points.shape[0], 1))
        points_homogeneous = np.hstack([moving_points, ones])
        # Step 2: Apply the 4x4 matrix using dot product (Nx4 @ 4x4 -> Nx4)
        transformed_homogeneous = points_homogeneous @ transformation_matrix.T

        # Step 3: Extract the first 3 columns back to Nx3
        transformed_points = np.round(transformed_homogeneous[:, :3])

        print(f'Transformed points in {self.fixed} space')
        for structure, point, f in zip(structures, transformed_points, fixed_points):
            print(structure, np.round(point/fixed_spacing), np.linalg.norm(f-point))
        #print(transformed_points/fixed_spacing)
        distances = np.linalg.norm(fixed_points-transformed_points, axis=1)
        print('Distances using umeyama:')
        mean = float(np.mean(distances))
        median = float(np.median(distances))
        maximum = float(np.max(distances))
        print(
            f"\tmean={mean:.3f} µm, "
            f"median={median:.3f} µm, "
            f"max={maximum:.3f} µm"
        )



                


    def register_downsampled(self):
        transforms = glob.glob(os.path.join(self.reg_path, f'{self.moving}_{self.fixed}_{self.transformation}_{self.downsample}_Composite*'))
        transforms = sorted(transforms)
        # 1. Load the 3D brain volumes
        if len(transforms) > 0:
            print(f"Transforms exist, exiting \n: {transforms}")
            exit(0)

        if not os.path.exists(self.fixed_image_path):
            print(f'Missing: {self.fixed_image_path}')
            exit(0)
        if not os.path.exists(self.moving_image_path):
            print(f'Missing: {self.moving_image_path}')
            exit(0)

        output_dir = Path(self.reg_path)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )



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
            write_composite_transform=False,
            verbose=self.debug
        )
        print(f'Finished registering')
        transform_list = reg['fwdtransforms']

        # Read the transform object and write it to your desired destination
        # For linear transforms (Rigid/Affine), there is one matrix file (.mat)
        # For SyN, you will have both a .mat and warp field(s) (.nii.gz)
        for i, path in enumerate(transform_list):
            print('path is', path)
            filename = os.path.basename(path)
            suffix = Path(filename).suffix
            newfilename = os.path.join(self.reg_path, f'{self.moving}_{self.fixed}_{self.transformation}_{self.downsample}_{i}{suffix}')
            shutil.move(path, newfilename)
            print('newfilename', newfilename)
            transforms.append(newfilename)            


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
    transformations = ['Affine', 'Rigid', 'SyN','ElasticSyN','SyNCC','Elastic']
    if transformation not in transformations:
        print(f'{transformation} is not in {transformations}')
        exit(0)

    pipeline = AntsRegistration(moving, fixed, downsample, transformation, debug=debug)


    function_mapping = {
        "register_volume": pipeline.register_downsampled,
        "register_points": pipeline.register_points,
        "testing": pipeline.testing
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')
