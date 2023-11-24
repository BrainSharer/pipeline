import SimpleITK as sitk
from pathlib import Path
import sys
import os


PIPELINE_ROOT = Path("./src").absolute()
sys.path.append(PIPELINE_ROOT.as_posix())
print(PIPELINE_ROOT)

from library.registration.section_registration import SectionRegistration

prep_id = 'DK78'
section = 181 # section = fixed_index
aligner = SectionRegistration(prep_id, section, debug=True)
print(f'fixed index={aligner.fixed_index}')
print(f'moving index={aligner.moving_index}')
elastixImageFilter = aligner.setup_registration()

elastixImageFilter.SetOutputDirectory(aligner.registration_output)
#elastixImageFilter.PrintParameterMap()
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



