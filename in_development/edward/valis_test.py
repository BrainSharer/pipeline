import os
import time
import numpy as np
from valis import registration
from valis.micro_rigid_registrar import MicroRigidRegistrar # For high resolution rigid registration


"""
To avoid this warning, please rebuild your copy of OpenBLAS with a larger NUM_THREADS setting
or set the environment variable OPENBLAS_NUM_THREADS to 128 or lower
"""
os.environ["OPENBLAS_NUM_THREADS"] = "1"

slide_src_dir = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/C1/thumbnail_cropped"
results_dst_dir = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/C1/valis"
os.makedirs(results_dst_dir, exist_ok=True)
registered_slide_dst_dir = "/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK37/preps/C1/thumbnail_registered"
os.makedirs(registered_slide_dst_dir, exist_ok=True)

micro_reg_fraction = 1/32

# Create a Valis object and use it to register the slides in slide_src_dir
#registrar = registration.Valis(slide_src_dir, results_dst_dir, imgs_ordered=True, non_rigid_registrar_cls=None)
#rigid_registrar, non_rigid_registrar, error_df = registrar.register()

ordered_img_list = []
files = sorted(os.listdir(slide_src_dir))
for file in files:
    filepath = os.path.join(slide_src_dir, file)
    ordered_img_list.append(filepath)

start = time.time()
registrar = registration.Valis(
    slide_src_dir,
    results_dst_dir,
    micro_rigid_registrar_cls=MicroRigidRegistrar,
    img_list=ordered_img_list,
    imgs_ordered=True,
    non_rigid_registrar_cls=None,
)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
print(f"Image dimensions: {img_dims}")
min_max_size = np.min([np.max(d) for d in img_dims])
print(f"Minimum maximum size: {min_max_size}")
img_areas = [np.multiply(*d) for d in img_dims]
max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)
# Perform high resolution non-rigid registration using 25% full resolution
micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)
registrar.warp_and_save_slides(registered_slide_dst_dir, crop="overlap")


registration.kill_jvm() # Kill the JVM
