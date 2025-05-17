"""
Important notes

If your fixed image has a smaller field of view than your moving image, 
your moving image will be cropped. (This is what happens when the brain stem
gets cropped out when setting a neurotrace brain against the Allen atlas. 
"""

import argparse
import sys
from pathlib import Path


PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.registration.volume_registration import VolumeRegistration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True, type=str)
    parser.add_argument("--channel", help="Enter channel", required=False, default=1, type=int)
    parser.add_argument('--um', help="size of atlas in micrometers", required=False, default=25, type=int)
    parser.add_argument('--scaling_factor', help="scaling factor to downsample", required=False, type=int)
    parser.add_argument('--fixed', help='Enter the fixed animal|atlas', required=False, type=str)
    parser.add_argument('--orientation', help='Enter the orientation: sagittal|coronal', required=False, default='sagittal', type=str)
    parser.add_argument("--bspline", help="Enter true or false", required=False, default="false", type=str)
    parser.add_argument("--task", help="Enter the task you want to perform", required=True, default="status", type=str)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    
    args = parser.parse_args()
    moving = args.moving
    channel = args.channel
    um = args.um
    scaling_factor = args.scaling_factor
    fixed = args.fixed
    orientation = args.orientation
    bspline = bool({"true": True, "false": False}[str(args.bspline).lower()])
    task = str(args.task).strip().lower()
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    volumeRegistration = VolumeRegistration(moving, channel, um, scaling_factor, fixed, orientation, bspline, debug)


    function_mapping = {'create_volume': volumeRegistration.create_volume,
                        'register_volume': volumeRegistration.register_volume,
                        'reverse_register_volume': volumeRegistration.reverse_register_volume,
                        'transformix_volume': volumeRegistration.transformix_volume,
                        'transformix_points': volumeRegistration.transformix_points,
                        'transformix_coms': volumeRegistration.transformix_coms,
                        'create_precomputed': volumeRegistration.create_precomputed,
                        'status': volumeRegistration.check_status,
                        'insert_points': volumeRegistration.insert_points,
                        'fill_contours': volumeRegistration.fill_contours,
                        'polygons': volumeRegistration.transformix_polygons,
                        'create_average_volume': volumeRegistration.create_average_volume,
                        'crop': volumeRegistration.crop_volume,
                        'origins': volumeRegistration.volume_origin_creation,
                        'transformix_origins': volumeRegistration.transformix_origins,
                        'pad_volume': volumeRegistration.pad_volume,
                        'group_volume': volumeRegistration.group_volume,
                        'create_brain_coms': volumeRegistration.create_brain_coms,
                        'create_moving_fixed_points': volumeRegistration.create_moving_fixed_points,
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')

