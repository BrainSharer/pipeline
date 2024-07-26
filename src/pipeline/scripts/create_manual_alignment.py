import argparse
from pathlib import Path
import sys

PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.registration.section_registration import SectionRegistration

section = 181 # section = fixed_index
def manual_alignment(animal, section, debug):
    aligner = SectionRegistration(animal, section, debug=debug)
    print(f'fixed index={aligner.fixed_index}')
    print(f'moving index={aligner.moving_index}')
    elastixImageFilter = aligner.setup_registration()

    elastixImageFilter.SetOutputDirectory(aligner.registration_output)
    if debug:
        elastixImageFilter.PrintParameterMap()
    elastixImageFilter.Execute()
    translations = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
    rigid = elastixImageFilter.GetTransformParameterMap()[1]["TransformParameters"]
    x1, y1 = translations
    R, x2, y2 = rigid
    x = float(x1) + float(x2)
    y = float(y1) + float(y2)
    print(f'rotation={R}')
    print(f'xshift={x}')
    print(f'yshift={y}')
    sql = f'UPDATE elastix_transformation SET rotation={R}, xshift={x}, yshift={y} WHERE section=\'{aligner.moving_index}\' and FK_prep_id=\'{prep_id}\';'
    print(sql)
    metric = aligner.getMetricValue()
    print(metric)

def allen_alignment(animal, debug):
    pass
    aligner = SectionRegistration(animal, section, debug=debug)
    print(f'fixed index={aligner.fixed_index}')
    print(f'moving index={aligner.moving_index}')
    elastixImageFilter = aligner.setup_registration()

    elastixImageFilter.SetOutputDirectory(aligner.registration_output)
    if debug:
        elastixImageFilter.PrintParameterMap()
    elastixImageFilter.Execute()
    translations = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
    rigid = elastixImageFilter.GetTransformParameterMap()[1]["TransformParameters"]
    x1, y1 = translations
    R, x2, y2 = rigid
    x = float(x1) + float(x2)
    y = float(y1) + float(y2)
    print(f'rotation={R}')
    print(f'xshift={x}')
    print(f'yshift={y}')
    sql = f'UPDATE elastix_transformation SET rotation={R}, xshift={x}, yshift={y} WHERE section=\'{aligner.moving_index}\' and FK_prep_id=\'{prep_id}\';'
    print(sql)
    metric = aligner.getMetricValue()
    print(metric)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal', required=True, type=str)
    parser.add_argument("--section", help="Enter the moving section", required=True, type=int)
    parser.add_argument("--debug", help="Enter true or false", required=False, default="false", type=str)
    
    args = parser.parse_args()
    animal = args.animal
    section = args.section
    debug = bool({"true": True, "false": False}[str(args.debug).lower()])
    manual_alignment(animal, section, debug)
