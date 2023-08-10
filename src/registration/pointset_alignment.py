import numpy as np
import SimpleITK as sitk
from registration.Registration import Registration


class PointSetAlignment(Registration):
    def __init__(self, fixed, moving):
        super().__init__()
        self.fixed = fixed
        self.moving = moving

    def calculate_transform(self):
        self.transform = sitk.LandmarkBasedTransformInitializer(self.transformation_type,
                                                                    list(self.fixed.flatten()), list(self.moving.flatten()))


class RigidPointSetAlignment(PointSetAlignment):
    def __init__(self, fixed, moving):
        super().__init__(fixed, moving)
        self.transformation_type = sitk.VersorRigid3DTransform()


class AffinePointSetAlignment(PointSetAlignment):
    def __init__(self, fixed, moving):
        super().__init__(fixed, moving)
        self.transformation_type = sitk.AffineTransform(3)


class SimilarityPointSetAlignment(PointSetAlignment):
    def __init__(self, fixed, moving):
        super().__init__(fixed, moving)
        self.transformation_type = sitk.Similarity3DTransform()


def get_shared_key_and_array(fixed_dictionary, moving_dictionary):
    shared_keys = set(fixed_dictionary.keys()).intersection(
        set(moving_dictionary.keys()))
    fixed = np.array([fixed_dictionary[key] for key in shared_keys])
    moving = np.array([moving_dictionary[key] for key in shared_keys])
    return fixed, moving, shared_keys


def get_rigid_aligned_points(fixed_dictionary, moving_dictionary):
    fixed, moving, shared_keys = get_shared_key_and_array(
        fixed_dictionary, moving_dictionary)
    rigid = RigidPointSetAlignment(fixed, moving)
    transform = rigid.get_transform
    print(f' transform is {transform}')
    transformed_point = rigid.get_transformed_moving_point()
    return dict(zip(shared_keys, transformed_point))


def get_affine_alignment_points(fixed_dictionary, moving_dictionary):
    fixed, moving, shared_keys = get_shared_key_and_array(
        fixed_dictionary, moving_dictionary)
    affine = AffinePointSetAlignment(fixed, moving)
    transformed_point = affine.get_transformed_moving_point()
    return dict(zip(shared_keys, transformed_point))
    

def get_similarity_alignment_points(fixed_dictionary, moving_dictionary):
    fixed, moving, shared_keys = get_shared_key_and_array(
        fixed_dictionary, moving_dictionary)
    affine = SimilarityPointSetAlignment(fixed, moving)
    transformed_point = affine.get_transformed_moving_point()
    return dict(zip(shared_keys, transformed_point))
