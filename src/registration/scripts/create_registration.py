"""
Important notes,
If your fixed image has a smaller field of view than your moving image, 
your moving image will be cropped. (This is what happens when the brain stem
gets cropped out. However, when the fixed image is bigger than the moving image, we get the following error:
Too many samples map outside moving image buffer. The moving image needs to be properly initialized.

In other words, only the part your moving image 
that overlap with the fixed image is included in your result image.
To warp the whole image, you can edit the size of the domain in the 
transform parameter map to match your moving image, and pass your moving image and 
the transform parameter map to sitk.Transformix().
10um allen is 1320x800
25um allen is 528x320
aligned volume @ 32 is 2047x1109 - unreg size matches allen10um
aligned volume @ 64 is 1024x555 - unreg size matches allen25um
aligned volume @ 128 is 512x278
aligned volume @50 is 1310x710
full aligned is 65500x35500
Need to scale a moving image as close as possible to the fixed image
COM info:
allen SC: (368, 62, 227)
pred  SC: 369, 64, 219
TODO, transform polygons in DB using the transformation below
TODO Mar 2025 - 
See how our structures fit on top of the Allen structure positions.
Try changing our atlas to the Allen COMs
Do an affine trans of our atlas to Allen
Also try registering to the Princeton mouse atlas
"""

import argparse
import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())


from library.registration.volume_registration import VolumeRegistration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--moving', help='Enter the animal (moving)', required=True)
    parser.add_argument("--channel", help="Enter channel", required=False, default=1, type=int)
    parser.add_argument('--um', help="size of atlas in micrometers", required=False, default=25, type=int)
    parser.add_argument('--fixed', help='Enter the fixed animal|atlas', required=False, default='Allen')
    parser.add_argument('--orientation', help='Enter the orientation: sagittal|coronal', required=False, default='sagittal')
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false")
    parser.add_argument("--bspline", help="Enter true or false", required=False, default="false")
    parser.add_argument("--task", help="Enter the task you want to perform", required=False, default="status", type=str)
    
    args = parser.parse_args()
    moving = args.moving
    channel = args.channel
    um = args.um
    fixed = args.fixed
    orientation = args.orientation
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    bspline = bool({"true": True, "false": False}[str(args.bspline).lower()])
    task = str(args.task).strip().lower()
    volumeRegistration = VolumeRegistration(moving, channel, um, fixed, orientation, bspline, debug)


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
    }

    if task in function_mapping:
        function_mapping[task]()
    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')

