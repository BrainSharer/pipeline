import numpy as np
import sys
from pathlib import Path

PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())

from library.atlas.atlas_utilities import list_coms


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps points A to points B.
    Input:
      A: Nx3 numpy array of source points
      B: Nx3 numpy array of destination points
    Returns:
      R: 3x3 rotation matrix
      t: 3x1 translation vector
      A_aligned: Nx3 transformed A points
      rms_error: root-mean-square deviation after alignment
    """
    assert A.shape == B.shape, "Input point sets must have the same shape"
    N = A.shape[0]

    # 1. Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 2. Center the points
    AA = A - centroid_A
    BB = B - centroid_B

    # 3. Compute covariance matrix
    H = AA.T @ BB

    # 4. Compute SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 5. Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 6. Compute translation
    t = centroid_B - R @ centroid_A

    # 7. Apply transform
    A_aligned = (R @ A.T).T + t

    # 8. Compute error
    errors = np.linalg.norm(A_aligned - B, axis=1)
    rms_error = np.sqrt(np.mean(errors**2))

    return R, t, A_aligned, rms_error


# Example usage
if __name__ == "__main__":
    # Example fiducials (N x 3 arrays)
    fixed = 'Allen'
    moving = 'AtlasV8'
    fixed_src = list_coms(fixed)
    moving_src = list_coms(moving)

    R, t, A_aligned, rms_error = best_fit_transform(fixed_src, moving_src)

    print("Estimated rotation matrix:\n", R)
    print("Estimated translation vector:\n", t)
    print("RMS alignment error:", rms_error)
