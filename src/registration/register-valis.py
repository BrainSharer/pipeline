import os
import argparse
from valis import registration, feature_detectors, non_rigid_registrars, affine_optimizer
# OME-NGFF file format

def run_valis(animal):

    ROOT = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data'

    slide_src_dir = os.path.join(ROOT, animal, 'preps/C1/thumbnail')
    results_dst_dir = os.path.join(ROOT, animal, 'preps/C1/registration_results')
    registered_slide_dst_dir = os.path.join(ROOT, animal, 'preps/C1/registered')

    os.makedirs(results_dst_dir, exist_ok=True)
    os.makedirs(registered_slide_dst_dir, exist_ok=True)


    # Select feature detector, affine optimizer, and non-rigid registration method.
    # Will use KAZE for feature detection and description
    # SimpleElastix will be used for non-rigid warping and affine optimization
    feature_detector_cls = feature_detectors.KazeFD
    non_rigid_registrar_cls = non_rigid_registrars.SimpleElastixWarper
    affine_optimizer_cls = affine_optimizer.AffineOptimizerMattesMI

    # Create a Valis object and use it to register the slides in slide_src_dir
    registrar = registration.Valis(slide_src_dir, 
                                   results_dst_dir,
                                feature_detector_cls=feature_detector_cls,
                                affine_optimizer_cls=affine_optimizer_cls,
                                non_rigid_registrar_cls=non_rigid_registrar_cls)

    registrar.imgs_ordered=True
    registrar.max_processed_image_dim_px=119

    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    
    registrar.warp_and_save_slides(registered_slide_dst_dir, crop="overlap")
    registration.kill_jvm() # Kill the JVM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work on Animal")
    parser.add_argument("--animal", help="Enter the animal", required=True)
    args = parser.parse_args()
    animal = args.animal
    run_valis(animal)
