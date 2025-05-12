"""This program will create everything.
The only required argument is the animal and task. By default it will work on channel=1
and downsample = True. Run them in this sequence for channel 1, when that is done, run
them again for the remaining channels and then for the full resolution version:

- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task status
- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task mask
- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task clean
- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task histogram
- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task align
- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task realign
- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task neuroglancer
- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task omezarr
- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task ng_preview

Explanation for the tasks:

- extract - Metadata from the CZI files is extracted and inserted into the database. \
    The TIFF files are first extracted from the CZI files at the standard downsampling factor, and then \
    later, the full resolution images are extracted. Web friendly PNG files are created from the TIFF files \
    for viewing in the portal. After this step, a user can verify the \
    database data and make any ordering, replacement or reproduction corrections.
- mask - Masks and normalized images are then created for the cleaning process. \
    A segmentation algorithmn is used to create initial masks for each image. These masks are used \
    to clean each channel of any unwanted slide debris. The user can peform QC on the masks \
    and makes sure they remove the debris and not the desired tissue.
- clean - After the masks are verified to be accurate, the final masks are created and then \
    the images are cleaned from the masks.
- histogram - Histograms showing the distribution of the image intensity levels are created \
    for all cleaned channel 1 sections.
- align - Section to section alignment with Elastix is then run on the cleaned and placed images using a rigid transformation. 
- realign - If the alignment needs improvement, the user can run the realign task to realign the images.
- neuroglancer - The final step is creating the Neuroglancer precomputed data from the aligned and cleaned images.
- omezarr - The final step is creating the OME-Zarr data from the aligned and cleaned images. You can run this instead of neuroglancer.
- ng_preview - This will create a preview of the brain images as Neuroglancer state and insert into database. \
It also creates symbolic links to image stacks on imageserv.dk.ucsd.edu

**Timing results**

- The processes that take the longest and need the most monitoring are, cleaning, aligning \
and creating the neuroglancer images. The number of workers must be set correctly \
otherwise the workstations will crash if the number of workers is too high. If the number \
of workers is too low, the processes take too long.
- Cleaning full resolution of 480 images on channel 1 on ratto took 5.5 hours
- Aligning full resolution of 480 images on channel 1 on ratto took 6.8 hours
- Running entire neuroglancer process on 480 images on channel 1 on ratto took 11.3 hours

**Human intervention is required at several points in the process**

- After create meta the user needs to check the database and verify the images \
are in the correct order and the images look good.
- After the first create mask method - the user needs to check the colored masks \
and possible dilate or crop them.
- After the alignment process - the user needs to verify the alignment looks good. \
Creating fiducials and then running the realing task will improve the alignment.

**Switching projection in Neuroglancer** 

- This switches the top left and bottom right quadrants. Place this JSON directly below the 'position' key:
    crossSectionOrientation: [0, -0.7071067690849304, 0, 0.7071067690849304],
- if you use omezarr, you will need to switch the projection in the viewer.
    crossSectionOrientation: [0, 0, 0, -1],
 

"""
import argparse
from pathlib import Path
import sys
from timeit import default_timer as timer


PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.pipeline_process import Pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True, type=str)
    parser.add_argument("--channel", help="Enter channel", required=False, default=1, type=int)
    parser.add_argument(
        "--downsample",
        help="Enter true or false",
        required=False,
        default="true",
        type=str,
    )
    parser.add_argument("--scaling_factor", help="Enter scaling_factor", required=False, default=32.0, type=float)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    parser.add_argument(
        "--task",
        help="Enter the task you want to perform: \
                        extract|mask|clean|histogram|align|extra_channel|neuroglancer|status",
        required=False,
        default="status",
        type=str,
    )

    args = parser.parse_args()

    animal = args.animal
    channel = args.channel
    downsample = bool({"true": True, "false": False}[str(args.downsample).lower()])
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    task = str(args.task).strip().lower()

    pipeline = Pipeline(
        animal,
        channel=channel,
        downsample=downsample,
        scaling_factor=args.scaling_factor,
        task=task,
        debug=debug,
    )

    function_mapping = {
        "extract": pipeline.extract,
        "mask": pipeline.mask,
        "clean": pipeline.clean,
        "histogram": pipeline.histogram,
        "align": pipeline.align,
        "realign": pipeline.realign,
        "affine": pipeline.affine_align,
        "extra_channel": pipeline.extra_channel,
        "neuroglancer": pipeline.neuroglancer,
        "omezarr": pipeline.omezarr,
        "omezarrinfo": pipeline.omezarr_info,
        "omezarr2tif": pipeline.omezarr2tif,
        "shell": pipeline.shell,
        "align_masks": pipeline.align_masks,
        "status": pipeline.check_status,
        "ng_preview": pipeline.ng_preview,
    }

    if task in function_mapping:
        start_time = timer()
        function_mapping[task]()
        end_time = timer()
        total_elapsed_time = round((end_time - start_time), 2)
        if total_elapsed_time >= 3600:
            hours = total_elapsed_time // 3600
            minutes = (total_elapsed_time % 3600) // 60
            time_out_msg = f'took {int(hours)} hour(s) and {int(minutes)} minute(s).'
        else:
            time_out_msg = f'took {total_elapsed_time} seconds.'

        print(f"{task} {time_out_msg}")
        sep = "*" * 40 + "\n"
        pipeline.fileLogger.logevent(f"{task} {time_out_msg}\n{sep}")
    else:
        print(f"{task} is not a correct task. Choose one of these:")
        for key in function_mapping.keys():
            print(f"\t{key}")