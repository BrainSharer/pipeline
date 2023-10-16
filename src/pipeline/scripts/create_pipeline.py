"""This program will create everything.
The only required argument is the animal and step. By default it will work on channel=1
and downsample = True. Run them in this sequence:

- python src/pipeline/scripts/create_pipeline.py --animal DKXX --task
- python src/pipeline/scripts/create_pipeline.py --animal DKXX channel 2|3 --task
- python src/pipeline/scripts/create_pipeline.py --animal DKXX channel 1 downsample false --task
- python src/pipeline/scripts/create_pipeline.py --animal DKXX channel 2|3 downsample false --task

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
- align - Section to section alignment with Elastix is then run on the cleaned images using a rigid transformation. 
- create_metrics - Each section to section alignment with Elastix is evaluated on the cleaned images. This data \
    is entered into the database for reference purposes.
- neuroglancer - The final step is creating the Neuroglancer precomputed data from the aligned and cleaned images.

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
increasing the step size will make the pipeline move forward in the process.

**Switching projection in Neuroglancer** 

- This switches the top left and bottom right quadrants. Place this JSON directly below the 'position' key:
- crossSectionOrientation: [0, -0.7071067690849304, 0, 0.7071067690849304],

"""
import argparse
from pathlib import Path
import sys
from timeit import default_timer as timer

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.pipeline_process import Pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True)
    parser.add_argument("--rescan_number", help="Enter rescan number, default is 0", required=False, default=0)
    parser.add_argument("--channel", help="Enter channel", required=False, default=1, type=int)
    parser.add_argument("--downsample", help="Enter true or false", required=False, default="true")
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false")
    parser.add_argument("--iterations", help="Enter iterations for alignment", required=False, default=2, type=int)
    parser.add_argument("--nomask", help="Do not create a mask. Use all the tissue.", 
                        required=False, default=False, type=str)
    parser.add_argument("--task", 
                        help="Enter the task you want to perform: \
                        extract|mask|clean|histogram|align|create_metrics|extra_channel|neuroglancer|check_status",
                        required=False, default="check_status", type=str)

    args = parser.parse_args()

    animal = args.animal
    rescan_number = int(args.rescan_number)
    channel = args.channel
    iterations = args.iterations
    downsample = bool({"true": True, "false": False}[str(args.downsample).lower()])
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    nomask = bool({"true": True, "false": False}[str(args.nomask).lower()])
    task = str(args.task).strip().lower()

    pipeline = Pipeline(animal, rescan_number=rescan_number, channel=channel, iterations=iterations, 
                        downsample=downsample, nomask=nomask, task=task, debug=False)

    function_mapping = {'extract': pipeline.extract,
                        'mask': pipeline.mask,
                        'clean': pipeline.clean,
                        'histogram': pipeline.histogram,
                        'align': pipeline.align,
                        'create_metrics': pipeline.create_metrics,
                        'extra_channel': pipeline.extra_channel,
                        'neuroglancer': pipeline.neuroglancer,
                        'cell_labels': pipeline.cell_labels,
                        'status': pipeline.check_status
    }

    if task in function_mapping:
        start_time = timer()
        pipeline.logevent(f"START  {str(task)}, downsample: {str(downsample)}")
        function_mapping[task]()
        end_time = timer()
        total_elapsed_time = round((end_time - start_time),2)
        print(f'{task} took {total_elapsed_time} seconds')
        sep = "*" * 40 + "\n"
        pipeline.logevent(f"{task} took {total_elapsed_time} seconds\n{sep}")

    else:
        print(f'{task} is not a correct task. Choose one of these:')
        for key in function_mapping.keys():
            print(f'\t{key}')


