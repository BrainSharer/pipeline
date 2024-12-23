"""
resolution_xyu: tuple, optional
    Physical size per pixel and the unit. If None (the default), these values will be determined for each slide using the slides' metadata. 
    If provided, this physical pixel sizes will be used for all of the slides. This option is available in case one cannot easily access to 
    the original slides, but does have the information on pixel's physical units.

slide_dims_dict_wh : dict, optional
    Key= slide/image file name, value= dimensions = [(width, height), (width, height), ...] for each level. If None (the default), the slide 
    dimensions will be pulled from the slides' metadata. If provided, those values will be overwritten. This option is available in case one 
    cannot easily access to the original slides, but does have the information on the slide dimensions.

max_image_dim_px : int, optional
    Maximum width or height of images that will be saved. This limit is mostly to keep memory in check.

max_processed_image_dim_px : int, optional
    Maximum width or height of processed images. An important parameter, as it determines the size of of the image in which features will 
    be detected and displacement fields computed.

max_non_rigid_registartion_dim_px : int, optional
     Maximum width or height of images used for non-rigid registration. Larger values may yeild more accurate results, at the expense of 
     speed and memory. There is also a practical limit, as the specified size may be too large to fit in memory.

"""
import argparse
import os
import time
import numpy as np
from valis import registration
from valis import registration, feature_detectors, non_rigid_registrars, affine_optimizer, serial_rigid


"""
To avoid this warning, please rebuild your copy of OpenBLAS with a larger NUM_THREADS setting
or set the environment variable OPENBLAS_NUM_THREADS to 128 or lower
"""

class ValisManager:
    def __init__(self, animal, debug):
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        self.slide_src_dir = f"/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1/thumbnail_cropped"
        self.results_dst_dir = f"/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1/valis"
        os.makedirs(self.results_dst_dir, exist_ok=True)
        self.registered_slide_dst_dir = f"/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/{animal}/preps/C1/thumbnail_registered"
        os.makedirs(self.registered_slide_dst_dir, exist_ok=True)
        files = sorted(os.listdir(self.slide_src_dir))
        len_files = len(files)
        self.midpoint = len_files // 2
        midfile = files[self.midpoint]
        self.ordered_img_list = []
        for file in files:
            filepath = os.path.join(self.slide_src_dir, file)
            self.ordered_img_list.append(filepath)

        self.reference_slide = os.path.join(self.slide_src_dir, midfile)

    def simple_reg(self):

        registrar = registration.Valis(self.slide_src_dir, dst_dir = self.results_dst_dir, non_rigid_registrar_cls=None,
            imgs_ordered=False,
            image_type="fluorescence",
            resolution_xyu=(10.4*2, 10.4*2, u'\u00B5m'),
            max_processed_image_dim_px=174,
            max_image_dim_px=174,
            align_to_reference=False,
)
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        #registrar.warp_and_merge_slides(self.registered_slide_dst_dir, crop = False, drop_duplicates = False )



    def register_to_mid(self):
        feature_detector_cls = feature_detectors.CensureVggFD
        non_rigid_registrar_cls = non_rigid_registrars.SimpleElastixWarper
        affine_optimizer_cls = affine_optimizer.AffineOptimizerMattesMI
        # Create a Valis object and use it to register the slides in slide_src_dir, aligning *towards* the reference slide.
        registrar = registration.Valis(
            self.slide_src_dir,
            self.results_dst_dir,
            feature_detector_cls=feature_detector_cls,
            img_list=self.ordered_img_list,
            imgs_ordered=True,
            image_type="fluorescence",
            resolution_xyu=(10.4*2, 10.4*2, u'\u00B5m'),
            max_processed_image_dim_px=119,
            max_image_dim_px=119,
            align_to_reference=False,
            non_rigid_registrar_cls=None)
        
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        registrar.warp_and_merge_slides(self.registered_slide_dst_dir, crop = False, drop_duplicates = False )




    def micro_reg(self):
        micro_reg_fraction = 1/32

        # Create a Valis object and use it to register the slides in slide_src_dir
        # registrar = registration.Valis(slide_src_dir, results_dst_dir, imgs_ordered=True, non_rigid_registrar_cls=None)
        # rigid_registrar, non_rigid_registrar, error_df = registrar.register()


        registrar = registration.Valis(
            self.slide_src_dir,
            self.results_dst_dir,
            micro_rigid_registrar_cls=MicroRigidRegistrar,
            img_list=self.ordered_img_list,
            imgs_ordered=True,
            image_type="fluorescence",
            non_rigid_registrar_cls=None,
            resolution_xyu=(10.4*2, 10.4*2, u'\u00B5m'),
            max_processed_image_dim_px=783,
            max_image_dim_px=783,
        )
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        rigid_registrar.save_displacement_fields(self.results_dst_dir)
        registrar.warp_and_merge_slides(self.registered_slide_dst_dir, crop = False, drop_duplicates = False )

        """
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
        """

    def teardown(self):
        registration.kill_jvm() # Kill the JVM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='Enter the animal animal', required=True)
    parser.add_argument('--debug', help='Enter debug True|False', required=False, default='true')

    args = parser.parse_args()
    animal = args.animal
    debug = bool({'true': True, 'false': False}[str(args.debug).lower()])
    valisManager = ValisManager(animal, debug)
    valisManager.simple_reg()
    #valisManager.teardown()
