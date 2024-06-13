import os
import sys
import SimpleITK as sitk
from pathlib import Path
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.image_manipulation.filelocation_manager import FileLocationManager
from library.utilities.utilities_registration import create_rigid_parameters


def lines_that_start_with(string, fp):
    return [line for line in fp if line.startswith(string)]

def getMetricValue(logpath):
    with open(logpath, "r") as fp:
        for line in lines_that_start_with("Final", fp):
            return line

def align_elastix(animal, fixed_file, moving_file, moving_index):
    pixelType = sitk.sitkFloat32
    fixed = sitk.ReadImage(fixed_file, pixelType)
    moving = sitk.ReadImage(moving_file, pixelType)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    rigid_params = elastixImageFilter.GetDefaultParameterMap("rigid")
    rigid_params["WriteResultImage"] = ["false"]
    elastixImageFilter.SetParameterMap(rigid_params)

    # logging
    fileLocationManager = FileLocationManager(animal)
    OUTPUT = fileLocationManager.elastix
    elastixImageFilter.LogToConsoleOff();
    elastixImageFilter.LogToFileOn();
    logfile = str(moving_index).zfill(3) + ".tif"
    os.makedirs(OUTPUT, exist_ok=True)
    elastixImageFilter.SetOutputDirectory(OUTPUT);
    elastixImageFilter.SetLogFileName(logfile);    
    elastixImageFilter.Execute()
    logfilepath = os.path.join(OUTPUT, logfile)
    metric_value = getMetricValue(logfilepath)
    print(metric_value)

def create_image(fixed_file, moving_file):
    elastixImageFilter = sitk.ElastixImageFilter()
    pixelType = sitk.sitkFloat32
    fixed = sitk.ReadImage(fixed_file, pixelType)
    moving = sitk.ReadImage(moving_file, pixelType)
    elastixImageFilter.SetFixedImage(fixed)
    elastixImageFilter.SetMovingImage(moving)
    rigid_params = create_rigid_parameters(elastixImageFilter)
    elastixImageFilter.SetParameterMap(rigid_params)
    elastixImageFilter.Execute()
    rigidTransform = elastixImageFilter.GetTransformParameterMap()[0]



    transformixFilter = sitk.TransformixImageFilter()
    transformixFilter.SetTransformParameterMap(rigidTransform)
    transformixFilter.SetMovingImage(moving)
    transformixFilter.Execute()
    transformed = transformixFilter.GetResultImage()
    transformed = sitk.Cast(transformed, sitk.sitkUInt16)
    print(f'new image type {type(transformed)}')
    sitk.WriteImage(transformed, 'xxx.tif')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Need animal fixed moving as arguments')
        exit(0)    
    animal = sys.argv[1]
    fixed = sys.argv[2]
    moving = sys.argv[3]
    moving_index = os.path.basename(moving)
    moving_index = str(moving_index).replace(".tif","")
    align_elastix(animal, fixed, moving, moving_index)
    #create_image(fixed, moving)
