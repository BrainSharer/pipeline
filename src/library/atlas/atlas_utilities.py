import numpy as np

def apply_affine_transform(point: list) -> np.ndarray:
    """
    Applies an affine transformation to a 3D point.

    Parameters:
    point (tuple or list): A tuple (x, y, z) representing the 3D point.
    matrix (numpy array): A 4x4 affine transformation matrix.

    Returns:
    numpy array: Transformed (x', y', z') coordinates as a numpy array.
    """
    matrix0 = np.array(
        [
            [9.36873602e-01, 6.25910930e-02, 3.41078823e-03, 4.07945327e02],
            [5.68396089e-04, 1.18742192e00, 6.28369930e-03, 4.01267566e01],
            [-1.27831427e-02, 8.42516452e-03, 1.11913658e00, -6.42895756e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    matrix1 = np.array(
        [
            [9.36873602e-01, 5.68396089e-04, -1.27831427e-02, 4.07945327e02],
            [6.25910930e-02, 1.18742192e00, 8.42516452e-03, 4.01267566e01],
            [3.41078823e-03, 6.28369930e-03, 1.11913658e00, -6.42895756e01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    matrix = matrix1

    if len(point) != 3:
        raise ValueError("Point must be a 3-element tuple or list (x, y, z)")

    if matrix.shape != (4, 4):
        raise ValueError("Matrix must be a 4x4 numpy array")

    # Convert the point to homogeneous coordinates
    homogeneous_point = np.array([point[0], point[1], point[2], 1])

    # Apply the transformation
    transformed_point = np.dot(matrix, homogeneous_point)

    # Return the transformed x, y, z coordinates (ignoring the homogeneous coordinate)
    return transformed_point[:3]
