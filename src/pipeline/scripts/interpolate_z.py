import numpy as np
from scipy.spatial import cKDTree

def match_points(sliceA, sliceB, max_dist=None):
    """
    Match points from sliceA to sliceB using nearest-neighbor matching.
    sliceA, sliceB: (N,2) arrays
    Returns list of tuples: [(iA, iB), ...]
    """

    if len(sliceA) == 0 or len(sliceB) == 0:
        return []

    tree = cKDTree(sliceB)
    dists, idxs = tree.query(sliceA)

    matches = []
    for iA, (d,iB) in enumerate(zip(dists, idxs)):
        if max_dist is None or d <= max_dist:
            matches.append((iA, iB))

    return matches


def interpolate_between_slices(z0, pts0, z1, pts1, num_interp=1):
    """
    Linearly interpolate point pairs between z0 and z1.
    Returns dict of z → array of interpolated points.
    """

    matches = match_points(pts0, pts1)

    interpolated = {}

    # Determine the z values to fill between
    z_vals = np.linspace(z0, z1, num_interp + 2)[1:-1]  # skip endpoints

    for z in z_vals:
        alpha = (z - z0) / (z1 - z0)
        interp_pts = []

        for i0, i1 in matches:
            p0 = pts0[i0]
            p1 = pts1[i1]
            pz = (1 - alpha) * p0 + alpha * p1
            interp_pts.append(pz)

        interpolated[z] = np.vstack(interp_pts) if interp_pts else np.zeros((0,2))

    return interpolated


def interpolate_all(points_by_z, num_interp=1):
    """
    points_by_z: dict {z : (N,2) array of points}
    num_interp: number of new slices to insert between each pair
    """
    result = {}
    zs = sorted(points_by_z.keys())

    for i in range(len(zs)-1):
        z0, z1 = zs[i], zs[i+1]
        pts0, pts1 = points_by_z[z0], points_by_z[z1]

        # keep original slice
        result[z0] = pts0

        # interpolate between slices
        inter = interpolate_between_slices(z0, pts0, z1, pts1, num_interp)
        result.update(inter)

    # also keep last slice
    result[zs[-1]] = points_by_z[zs[-1]]

    return result


points_by_z = {
    0: np.array([[10,10],[50,20],[30,40]]),
    5: np.array([[12,12],[48,22]]),
    10: np.array([[14,14],[45,25],[28,38],[60,50]])
}

filled = interpolate_all(points_by_z, num_interp=2)

for z in sorted(filled.keys()):
    print("z =", z)
    print(filled[z])