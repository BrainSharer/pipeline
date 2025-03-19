import argparse
import os
import sys
import numpy as np

from pathlib import Path

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

try:
    from settings import data_path, host, schema
except ImportError:
    print('Missing settings using defaults')
    data_path = "/net/birdstore/Active_Atlas_Data/data_root"
    host = "db.dk.ucsd.edu"
    schema = "brainsharer"

from library.controller.sql_controller import SqlController
from library.image_manipulation.filelocation_manager import FileLocationManager


def parameter_elastix_parameter_file_to_dict(filename):
    d = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('('):
                tokens = line[1:-2].split(' ')
                key = tokens[0]
                if len(tokens) > 2:
                    value = []
                    for v in tokens[1:]:
                        try:
                            value.append(float(v))
                        except ValueError:
                            value.append(v)
                else:
                    v = tokens[1]
                    try:
                        value = (float(v))
                    except ValueError:
                        value = v
                d[key] = value

        return d

def parse_elastix_parameter_file(filepath):
    """
    Parse elastix parameter result file.
    """
    
    d = parameter_elastix_parameter_file_to_dict(filepath)
    
    # For alignment composition script
    rot_rad, x_mm, y_mm = d['TransformParameters']
    center = np.array(d['CenterOfRotationPoint']) / np.array(d['Spacing'])
    # center[1] = d['Size'][1] - center[1]

    xshift = x_mm / d['Spacing'][0]
    yshift = y_mm / d['Spacing'][1]

    R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                    [np.sin(rot_rad), np.cos(rot_rad)]])
    shift = center + (xshift, yshift) - np.dot(R, center)
    T = np.vstack([np.column_stack([R, shift]), [0,0,1]])
    return T


def slurp(animal):
    sqlController = SqlController(animal)
    fileLocationManager = FileLocationManager(animal)


    INPUT = os.path.join(fileLocationManager.prep, 'C1', 'thumbnail_cleaned')
    if not os.path.exists(INPUT):
        print(f'{INPUT} does not exist')
        sys.exit()
    ELASTIX = fileLocationManager.elastix
    files = sorted(os.listdir(INPUT))
    for i in range(1, len(files)):
        fixed_index = os.path.splitext(files[i - 1])[0]
        moving_index = os.path.splitext(files[i])[0]

        parse_dir = '{}_to_{}'.format(moving_index, fixed_index)
        output_subdir = os.path.join(ELASTIX, parse_dir)
        filepath = os.path.join(output_subdir, 'TransformParameters.0.txt')

        if os.path.exists(filepath):
            d = parameter_elastix_parameter_file_to_dict(filepath)
            rotation, xshift, yshift = d['TransformParameters']
            T = parse_elastix_parameter_file(filepath)
            print(f'{filepath} rotation={rotation} xshift={xshift}, yshift={yshift} T={T.shape}')
            #animal, section, rotation, xshift, yshift, metric, iteration
            sqlController.add_elastix_row(animal, moving_index, rotation, xshift, yshift, 0, 0)
        else:
            print(f'{filepath} does not exist')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter animal', required=True)    
    args = parser.parse_args()
    animal = args.animal
    slurp(animal)    

