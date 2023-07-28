from pathlib import Path

import SimpleITK as sitk

def load_image_dir(image_dir, spacing=None):
    """A helper function to load a directory of images as a SimpleITK Image."""
    image_dir = Path(image_dir).resolve()
    image_series = []
    for image_file in sorted(image_dir.iterdir()):
        print(f'Loading image {image_file.name}', end='\r')
        image = sitk.ReadImage(image_file.as_posix())
        image_series.append(image)
    sitk_image = sitk.JoinSeries(image_series)
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    return sitk_image
