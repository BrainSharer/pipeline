"""Some registration-related algorithms."""
import numpy as np


def umeyama(src, dst, with_scaling=True):
    """The Umeyama algorithm to register landmarks with rigid transform.

    See the paper "Least-squares estimation of transformation parameters
    between two point patterns".
    """
    src = np.array(src)
    dst = np.array(dst)
    assert src.shape == dst.shape, 'Shapes of destination and sources are not the same'
    assert len(src.shape) == 2, 'Shapes of destination and sources are not two dimensional'
    assert src.shape[1] > 3, 'There are not enough points to work with in umeyama'
    m, n = src.shape

    src_mean = np.mean(src, axis=1).reshape(-1, 1)
    dst_mean = np.mean(dst, axis=1).reshape(-1, 1)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    u, s, vh = np.linalg.svd(dst_demean @ src_demean.T / n)

    # deal with reflection
    e = np.ones(m)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        e[-1] = -1

    r = u @ np.diag(e) @ vh

    if with_scaling:
        src_var = (src_demean ** 2).sum(axis=0).mean()
        c = sum(s * e) / src_var
        r *= c

    t = dst_mean - r @ src_mean
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        print(r,t)
        import sys
        sys.exit()

    return r, t



def atlas_to_brain_transform(atlas_coord, r, t):
    """
    Takes an x,y,z brain coordinates, and a rotation matrix and translation vector.
    Returns the point in atlas coordinates in micrometers.
    params:
        atlas_coord: tuple of x,y,z coordinates of the atlas in micrometers
        r: float of the rotation matrix
        t: vector of the translation matrix
    """
    # Bring atlas coordinates to physical space
    atlas_coord = np.array(atlas_coord).reshape(3, 1) # Convert to a column vector
    # Apply affine transformation in physical space
    r_inv = np.linalg.inv(r)
    brain_coord_phys = r_inv @ atlas_coord - (r_inv @  t)
    # Bring brain coordinates back to brain space
    return brain_coord_phys.T[0] # Convert back to a row vector


def brain_to_atlas_transform(brain_coord, r, t):
    """
    Takes an x,y,z brain coordinates, and a rotation matrix and translation vector.
    params:
        atlas_coord: tuple of x,y,z coordinates of the atlas in micrometers
        r: float of the rotation matrix
        t: vector of the translation matrix
    Returns the point in atlas coordinates in micrometers.
    """
    # Transform brain coordinates to physical space
    brain_coord = np.array(brain_coord).reshape(3, 1) # Convert to a column vector
    atlas_coord = r @ brain_coord + t
    return atlas_coord.T[0] # Convert back to a row vector

def align_point_dictionary(moving_points,fixed_points):
    common_keys = moving_points.keys() & fixed_points.keys()
    fixed_point_set = np.array([fixed_points[s] for s in common_keys]).T
    moving_point_set = np.array([moving_points[s] for s in common_keys]).T
    R, t = umeyama(moving_point_set, fixed_point_set)
    return R,t